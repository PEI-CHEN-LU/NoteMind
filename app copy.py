import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from MilvusController import upload_file_in_milvus, search_similar_embeddings, delete_vector
from LLM import ask_LLM

# Set up logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
CORS(app)

# File upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'md', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Database configuration
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///notebooklm.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

db = SQLAlchemy(model_class=Base)
db.init_app(app)

# Login manager setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = '請先登入以存取此頁面'

# Database Models
class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)

    # Relationship to topics
    topics = db.relationship('Topic', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        """設置密碼雜湊"""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """檢查密碼"""
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

class Topic(db.Model):
    __tablename__ = 'topics'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    emoji = db.Column(db.String(10), default='📝')
    description = db.Column(db.Text)
    date = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    # Foreign key to user
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    # Relationship to content items
    content_items = db.relationship('ContentItem', backref='topic', lazy=True, cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Topic {self.title}>'

class ContentItem(db.Model):
    __tablename__ = 'content_items'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content_type = db.Column(db.String(50), nullable=False)  # 'file', 'note', 'link'
    content = db.Column(db.Text)  # For notes or links
    created_at = db.Column(db.DateTime, default=datetime.now)

    # Foreign key to topic
    topic_id = db.Column(db.Integer, db.ForeignKey('topics.id'), nullable=False)

    def __repr__(self):
        return f'<ContentItem {self.title}>'


class FileItem(db.Model):
    __tablename__ = 'files'

    id = db.Column(db.Integer, primary_key=True)
    file_path = db.Column(db.Text, nullable=False)
    file_name = db.Column(db.String(255), nullable=False)
    original_name = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer)
    mime_type = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    topic_id = db.Column(db.Integer, nullable=False)
    # Foreign key to user
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    def __repr__(self):
        return f'<FileItem {self.file_name}>'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create tables
with app.app_context():
    db.create_all()
    logging.info("Database tables created")

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash('請輸入使用者名稱和密碼', 'error')
            return render_template('auth/login.html')

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            flash(f'歡迎回來，{user.username}！', 'success')
            return redirect(next_page or url_for('index'))
        else:
            flash('使用者名稱或密碼錯誤', 'error')

    return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        # Validation
        if not all([username, email, password, confirm_password]):
            flash('請填寫所有欄位', 'error')
            return render_template('auth/register.html')

        if password != confirm_password:
            flash('密碼確認不符', 'error')
            return render_template('auth/register.html')

        if len(password) < 6:
            flash('密碼長度至少需要6個字元', 'error')
            return render_template('auth/register.html')

        # Check if user exists
        if User.query.filter_by(username=username).first():
            flash('使用者名稱已存在', 'error')
            return render_template('auth/register.html')

        if User.query.filter_by(email=email).first():
            flash('電子郵件已被註冊', 'error')
            return render_template('auth/register.html')

        # Create new user
        try:
            user = User(username=username, email=email)
            user.set_password(password)

            db.session.add(user)
            db.session.commit()

            flash('註冊成功！請登入', 'success')
            return redirect(url_for('login'))

        except Exception as e:
            db.session.rollback()
            error_msg = str(e)
            logging.error(f"Registration error: {error_msg}")

            if 'unique constraint' in error_msg.lower():
                if 'username' in error_msg.lower():
                    flash('使用者名稱已存在', 'error')
                elif 'email' in error_msg.lower():
                    flash('電子郵件已被註冊', 'error')
                else:
                    flash('使用者名稱或電子郵件已存在', 'error')
            else:
                flash('註冊失敗，請稍後再試', 'error')

            return render_template('auth/register.html')

    return render_template('auth/register.html')

@app.route('/logout')
@login_required
def logout():
    """Logout user"""
    logout_user()
    flash('已成功登出', 'success')
    return redirect(url_for('login'))

# Main routes
@app.route('/')
@login_required
def index():
    """Main page showing all topics in grid layout"""
    topics = Topic.query.filter_by(user_id=current_user.id).order_by(Topic.created_at.desc()).all()
    return render_template('index.html', topics=topics)

@app.route('/topic/<int:topic_id>')
@login_required
def topic_detail(topic_id):
    """Topic detail page"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()
    if not topic:
        flash('主題不存在', 'error')
        return redirect(url_for('index'))

    # Get files for this user (not topic-specific, all user files)
    files = FileItem.query.filter_by(user_id=current_user.id, topic_id=topic_id).order_by(FileItem.created_at.desc()).all()

    return render_template('topic_detail.html', topic=topic, files=files)

@app.route('/add_topic', methods=['GET', 'POST'])
@login_required
def add_topic():
    """Add new topic"""
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        emoji = request.form.get('emoji', '📝').strip()
        description = request.form.get('description', '').strip()

        if not title:
            flash('請輸入主題標題', 'error')
            return redirect(url_for('index'))

        new_topic = Topic(
            title=title,
            emoji=emoji if emoji else '📝',
            description=description if description else '新增的主題',
            date=datetime.now().strftime('%Y年%m月%d日'),
            user_id=current_user.id
        )

        try:
            db.session.add(new_topic)
            db.session.commit()
            flash('主題新增成功！', 'success')
            
        except Exception as e:
            db.session.rollback()
            flash('新增失敗，請稍後再試', 'error')
            logging.error(f"Add topic error: {e}")

        return redirect(url_for('index'))

    return redirect(url_for('index'))

@app.route('/delete_topic/<int:topic_id>', methods=['POST'])
@login_required
def delete_topic(topic_id):
    """Delete a topic"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()

    if not topic:
        return jsonify({'success': False, 'error': '主題不存在'})

    try:
        db.session.delete(topic)
        db.session.commit()
        
        # 刪除 milvus 內的相關資料 (current_user.id, topic_id) 
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        logging.error(f"Delete topic error: {e}")
        return jsonify({'success': False, 'error': '刪除失敗'})

@app.route('/delete_file/<int:file_id>', methods=['POST'])
@login_required
def delete_file(file_id):
    """Delete a file"""
    topic_id = request.args.get('topic_id')
    file_item = FileItem.query.filter_by(
        id=file_id,
        user_id=current_user.id
    ).first()

    if not file_item:
        return jsonify({'success': False, 'error': '檔案不存在'})

    try:
        # Delete the physical file first
        if os.path.exists(file_item.file_path):
            try:
                os.remove(file_item.file_path)
                logging.info(f"Successfully deleted file: {file_item.file_path}")
            except Exception as file_error:
                logging.error(f"Failed to delete file {file_item.file_path}: {file_error}")
                return jsonify({'success': False, 'error': '檔案刪除失敗'})
        else:
            logging.warning(f"File not found: {file_item.file_path}")

        # Delete the FileItem from database
        db.session.delete(file_item)
        db.session.commit()
        
        delete_vector(current_user.id, topic_id, file_id)
        # 刪除 milvus 內的相關資料 (current_user.id, topic_id, file_id)
        
        

        logging.info(f"Successfully deleted file item: {file_item.original_name}")
        return jsonify({'success': True})

    except Exception as e:
        db.session.rollback()
        logging.error(f"Delete file error: {e}")
        return jsonify({'success': False, 'error': '刪除失敗'})

@app.route('/add_file/<int:topic_id>', methods=['POST'])
@login_required
def add_file(topic_id):
    """Add file to a topic - only creates FileItem record, not ContentItem"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()
    if not topic:
        flash('主題不存在', 'error')
        return redirect(url_for('index'))

    # Handle file upload
    file = request.files.get('file')
    if not file or not file.filename:
        flash('請選擇檔案', 'error')
        return redirect(url_for('topic_detail', topic_id=topic_id))

    if not allowed_file(file.filename):
        flash('不支援的檔案格式', 'error')
        return redirect(url_for('topic_detail', topic_id=topic_id))

    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        safe_filename = timestamp + filename
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
        if not os.path.exists(folder_path):
            # 如果不存在，就建立資料夾
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, safe_filename)

        # Save the file
        file.save(file_path)

        # Create FileItem record only
        file_item = FileItem(
            file_path=file_path,
            file_name=safe_filename,
            original_name=file.filename,
            file_size=os.path.getsize(file_path),
            mime_type=file.content_type,
            user_id=current_user.id,
            topic_id=topic_id
        )
        db.session.add(file_item)
        db.session.commit()

        flash('檔案上傳成功！', 'success')
        upload_file_in_milvus(int(current_user.id), int(topic_id), int(file_item.id), file_path)
        # 切分檔案，上傳到 milvus (current_user.id, topic_id, file_id, file_path)

    except Exception as e:
        db.session.rollback()
        logging.error(f"File upload error: {e}")
        flash('檔案上傳失敗，請稍後再試', 'error')

        # Clean up file if database operation failed
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

    return redirect(url_for('topic_detail', topic_id=topic_id))

@app.route('/download/<int:file_id>')
@login_required
def download_file(file_id):
    """Download a file"""
    file_item = FileItem.query.filter_by(
        id=file_id,
        user_id=current_user.id
    ).first()

    if not file_item:
        flash('檔案不存在', 'error')
        return redirect(url_for('index'))

    if not os.path.exists(file_item.file_path):
        flash('檔案已被移動或刪除', 'error')
        return redirect(url_for('index'))

    directory = os.path.dirname(file_item.file_path)
    filename = os.path.basename(file_item.file_path)

    return send_from_directory(directory, filename, as_attachment=True, download_name=file_item.original_name)

@app.route('/save_notes/<int:topic_id>', methods=['POST'])
@login_required
def save_notes(topic_id):
    """Save notes for a topic"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()
    if not topic:
        return jsonify({'success': False, 'error': '主題不存在'})

    try:
        data = request.get_json()
        notes_content = data.get('notes', '')
        
        # 更新主題的描述欄位來保存筆記
        topic.description = notes_content
        topic.updated_at = datetime.now()
        
        db.session.commit()
        return jsonify({'success': True})
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Save notes error: {e}")
        return jsonify({'success': False, 'error': '保存失敗'})

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    if current_user.is_authenticated:
        topics = Topic.query.filter_by(user_id=current_user.id).all()
        return render_template('index.html', topics=topics), 404
    else:
        return redirect(url_for('login')), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors"""
    logging.error(f"500 error: {e}")
    db.session.rollback()

    if current_user.is_authenticated:
        flash('發生內部錯誤，請稍後再試', 'error')
        return redirect(url_for('index'))
    else:
        flash('發生內部錯誤，請稍後再試', 'error')
        return redirect(url_for('login'))
    
    
@app.route('/process_selected_files', methods=['POST'])
def process_selected_files():
    data = request.get_json()
    file_ids = data.get('fileIds', [])
    print(file_ids)
    # 在這裡進行你對選中檔案的處理
    # ...
    return jsonify({'success': True})


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    file_ids = data.get('fileIds', [])
    topic_id = data.get('topicId', '')
    question = data.get('question', '')
    print(data)
    response_msg = ''

    if file_ids == []:
        response_msg = '請至少選擇一項參考來源'
        return jsonify({'success': True, 'ai_answer': response_msg})
    
    if question == '':
        response_msg = '您想問甚麼呢?'
        return jsonify({'success': True, 'ai_answer': response_msg})
    
    file_id_list = [int(file_id) for file_id in file_ids]
    
    ref_info = search_similar_embeddings(int(current_user.id), int(topic_id), file_id_list, question)

    if ref_info == []:
        response_msg = '選取檔案內沒有相關內容'
        return jsonify({'success': True, 'ai_answer': response_msg})
    
    response_msg = ask_LLM(ref_info, question)
    # 在這裡進行你對選中檔案的處理
    # ...
    return jsonify({'success': True, 'ai_answer': response_msg.replace('\n','<br>')})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from MilvusController import upload_file_in_milvus, search_similar_embeddings, delete_vector
from LLM import ask_LLM

# Set up logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
CORS(app)

# File upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'md', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Database configuration
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///notebooklm.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

db = SQLAlchemy(model_class=Base)
db.init_app(app)

# Login manager setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = '請先登入以存取此頁面'

# Database Models
class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)

    # Relationship to topics
    topics = db.relationship('Topic', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        """設置密碼雜湊"""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """檢查密碼"""
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

class Topic(db.Model):
    __tablename__ = 'topics'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    emoji = db.Column(db.String(10), default='📝')
    description = db.Column(db.Text)
    date = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    # Foreign key to user
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    # Relationship to content items
    content_items = db.relationship('ContentItem', backref='topic', lazy=True, cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Topic {self.title}>'

class ContentItem(db.Model):
    __tablename__ = 'content_items'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content_type = db.Column(db.String(50), nullable=False)  # 'file', 'note', 'link'
    content = db.Column(db.Text)  # For notes or links
    created_at = db.Column(db.DateTime, default=datetime.now)

    # Foreign key to topic
    topic_id = db.Column(db.Integer, db.ForeignKey('topics.id'), nullable=False)

    def __repr__(self):
        return f'<ContentItem {self.title}>'


class FileItem(db.Model):
    __tablename__ = 'files'

    id = db.Column(db.Integer, primary_key=True)
    file_path = db.Column(db.Text, nullable=False)
    file_name = db.Column(db.String(255), nullable=False)
    original_name = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer)
    mime_type = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    topic_id = db.Column(db.Integer, nullable=False)
    # Foreign key to user
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    def __repr__(self):
        return f'<FileItem {self.file_name}>'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create tables
with app.app_context():
    db.create_all()
    logging.info("Database tables created")

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash('請輸入使用者名稱和密碼', 'error')
            return render_template('auth/login.html')

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            flash(f'歡迎回來，{user.username}！', 'success')
            return redirect(next_page or url_for('index'))
        else:
            flash('使用者名稱或密碼錯誤', 'error')

    return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        # Validation
        if not all([username, email, password, confirm_password]):
            flash('請填寫所有欄位', 'error')
            return render_template('auth/register.html')

        if password != confirm_password:
            flash('密碼確認不符', 'error')
            return render_template('auth/register.html')

        if len(password) < 6:
            flash('密碼長度至少需要6個字元', 'error')
            return render_template('auth/register.html')

        # Check if user exists
        if User.query.filter_by(username=username).first():
            flash('使用者名稱已存在', 'error')
            return render_template('auth/register.html')

        if User.query.filter_by(email=email).first():
            flash('電子郵件已被註冊', 'error')
            return render_template('auth/register.html')

        # Create new user
        try:
            user = User(username=username, email=email)
            user.set_password(password)

            db.session.add(user)
            db.session.commit()

            flash('註冊成功！請登入', 'success')
            return redirect(url_for('login'))

        except Exception as e:
            db.session.rollback()
            error_msg = str(e)
            logging.error(f"Registration error: {error_msg}")

            if 'unique constraint' in error_msg.lower():
                if 'username' in error_msg.lower():
                    flash('使用者名稱已存在', 'error')
                elif 'email' in error_msg.lower():
                    flash('電子郵件已被註冊', 'error')
                else:
                    flash('使用者名稱或電子郵件已存在', 'error')
            else:
                flash('註冊失敗，請稍後再試', 'error')

            return render_template('auth/register.html')

    return render_template('auth/register.html')

@app.route('/logout')
@login_required
def logout():
    """Logout user"""
    logout_user()
    flash('已成功登出', 'success')
    return redirect(url_for('login'))

# Main routes
@app.route('/')
@login_required
def index():
    """Main page showing all topics in grid layout"""
    topics = Topic.query.filter_by(user_id=current_user.id).order_by(Topic.created_at.desc()).all()
    return render_template('index.html', topics=topics)

@app.route('/topic/<int:topic_id>')
@login_required
def topic_detail(topic_id):
    """Topic detail page"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()
    if not topic:
        flash('主題不存在', 'error')
        return redirect(url_for('index'))

    # Get files for this user (not topic-specific, all user files)
    files = FileItem.query.filter_by(user_id=current_user.id, topic_id=topic_id).order_by(FileItem.created_at.desc()).all()

    return render_template('topic_detail.html', topic=topic, files=files)

@app.route('/add_topic', methods=['GET', 'POST'])
@login_required
def add_topic():
    """Add new topic"""
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        emoji = request.form.get('emoji', '📝').strip()
        description = request.form.get('description', '').strip()

        if not title:
            flash('請輸入主題標題', 'error')
            return redirect(url_for('index'))

        new_topic = Topic(
            title=title,
            emoji=emoji if emoji else '📝',
            description=description if description else '新增的主題',
            date=datetime.now().strftime('%Y年%m月%d日'),
            user_id=current_user.id
        )

        try:
            db.session.add(new_topic)
            db.session.commit()
            flash('主題新增成功！', 'success')
            
        except Exception as e:
            db.session.rollback()
            flash('新增失敗，請稍後再試', 'error')
            logging.error(f"Add topic error: {e}")

        return redirect(url_for('index'))

    return redirect(url_for('index'))

@app.route('/delete_topic/<int:topic_id>', methods=['POST'])
@login_required
def delete_topic(topic_id):
    """Delete a topic"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()

    if not topic:
        return jsonify({'success': False, 'error': '主題不存在'})

    try:
        db.session.delete(topic)
        db.session.commit()
        
        # 刪除 milvus 內的相關資料 (current_user.id, topic_id) 
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        logging.error(f"Delete topic error: {e}")
        return jsonify({'success': False, 'error': '刪除失敗'})

@app.route('/delete_file/<int:file_id>', methods=['POST'])
@login_required
def delete_file(file_id):
    """Delete a file"""
    topic_id = request.args.get('topic_id')
    file_item = FileItem.query.filter_by(
        id=file_id,
        user_id=current_user.id
    ).first()

    if not file_item:
        return jsonify({'success': False, 'error': '檔案不存在'})

    try:
        # Delete the physical file first
        if os.path.exists(file_item.file_path):
            try:
                os.remove(file_item.file_path)
                logging.info(f"Successfully deleted file: {file_item.file_path}")
            except Exception as file_error:
                logging.error(f"Failed to delete file {file_item.file_path}: {file_error}")
                return jsonify({'success': False, 'error': '檔案刪除失敗'})
        else:
            logging.warning(f"File not found: {file_item.file_path}")

        # Delete the FileItem from database
        db.session.delete(file_item)
        db.session.commit()
        
        delete_vector(current_user.id, topic_id, file_id)
        # 刪除 milvus 內的相關資料 (current_user.id, topic_id, file_id)
        
        

        logging.info(f"Successfully deleted file item: {file_item.original_name}")
        return jsonify({'success': True})

    except Exception as e:
        db.session.rollback()
        logging.error(f"Delete file error: {e}")
        return jsonify({'success': False, 'error': '刪除失敗'})

@app.route('/add_file/<int:topic_id>', methods=['POST'])
@login_required
def add_file(topic_id):
    """Add file to a topic - only creates FileItem record, not ContentItem"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()
    if not topic:
        flash('主題不存在', 'error')
        return redirect(url_for('index'))

    # Handle file upload
    file = request.files.get('file')
    if not file or not file.filename:
        flash('請選擇檔案', 'error')
        return redirect(url_for('topic_detail', topic_id=topic_id))

    if not allowed_file(file.filename):
        flash('不支援的檔案格式', 'error')
        return redirect(url_for('topic_detail', topic_id=topic_id))

    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        safe_filename = timestamp + filename
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
        if not os.path.exists(folder_path):
            # 如果不存在，就建立資料夾
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, safe_filename)

        # Save the file
        file.save(file_path)

        # Create FileItem record only
        file_item = FileItem(
            file_path=file_path,
            file_name=safe_filename,
            original_name=file.filename,
            file_size=os.path.getsize(file_path),
            mime_type=file.content_type,
            user_id=current_user.id,
            topic_id=topic_id
        )
        db.session.add(file_item)
        db.session.commit()

        flash('檔案上傳成功！', 'success')
        upload_file_in_milvus(int(current_user.id), int(topic_id), int(file_item.id), file_path)
        # 切分檔案，上傳到 milvus (current_user.id, topic_id, file_id, file_path)

    except Exception as e:
        db.session.rollback()
        logging.error(f"File upload error: {e}")
        flash('檔案上傳失敗，請稍後再試', 'error')

        # Clean up file if database operation failed
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

    return redirect(url_for('topic_detail', topic_id=topic_id))

@app.route('/download/<int:file_id>')
@login_required
def download_file(file_id):
    """Download a file"""
    file_item = FileItem.query.filter_by(
        id=file_id,
        user_id=current_user.id
    ).first()

    if not file_item:
        flash('檔案不存在', 'error')
        return redirect(url_for('index'))

    if not os.path.exists(file_item.file_path):
        flash('檔案已被移動或刪除', 'error')
        return redirect(url_for('index'))

    directory = os.path.dirname(file_item.file_path)
    filename = os.path.basename(file_item.file_path)

    return send_from_directory(directory, filename, as_attachment=True, download_name=file_item.original_name)

@app.route('/save_notes/<int:topic_id>', methods=['POST'])
@login_required
def save_notes(topic_id):
    """Save notes for a topic"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()
    if not topic:
        return jsonify({'success': False, 'error': '主題不存在'})

    try:
        data = request.get_json()
        notes_content = data.get('notes', '')
        
        # 更新主題的描述欄位來保存筆記
        topic.description = notes_content
        topic.updated_at = datetime.now()
        
        db.session.commit()
        return jsonify({'success': True})
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Save notes error: {e}")
        return jsonify({'success': False, 'error': '保存失敗'})

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    if current_user.is_authenticated:
        topics = Topic.query.filter_by(user_id=current_user.id).all()
        return render_template('index.html', topics=topics), 404
    else:
        return redirect(url_for('login')), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors"""
    logging.error(f"500 error: {e}")
    db.session.rollback()

    if current_user.is_authenticated:
        flash('發生內部錯誤，請稍後再試', 'error')
        return redirect(url_for('index'))
    else:
        flash('發生內部錯誤，請稍後再試', 'error')
        return redirect(url_for('login'))
    
    
@app.route('/process_selected_files', methods=['POST'])
def process_selected_files():
    data = request.get_json()
    file_ids = data.get('fileIds', [])
    print(file_ids)
    # 在這裡進行你對選中檔案的處理
    # ...
    return jsonify({'success': True})


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    file_ids = data.get('fileIds', [])
    topic_id = data.get('topicId', '')
    question = data.get('question', '')
    print(data)
    response_msg = ''

    if file_ids == []:
        response_msg = '請至少選擇一項參考來源'
        return jsonify({'success': True, 'ai_answer': response_msg})
    
    if question == '':
        response_msg = '您想問甚麼呢?'
        return jsonify({'success': True, 'ai_answer': response_msg})
    
    file_id_list = [int(file_id) for file_id in file_ids]
    
    ref_info = search_similar_embeddings(int(current_user.id), int(topic_id), file_id_list, question)

    if ref_info == []:
        response_msg = '選取檔案內沒有相關內容'
        return jsonify({'success': True, 'ai_answer': response_msg})
    
    response_msg = ask_LLM(ref_info, question)
    # 在這裡進行你對選中檔案的處理
    # ...
    return jsonify({'success': True, 'ai_answer': response_msg.replace('\n','<br>')})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from MilvusController import upload_file_in_milvus, search_similar_embeddings, delete_vector
from LLM import ask_LLM

# Set up logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
CORS(app)

# File upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'md', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Database configuration
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///notebooklm.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

db = SQLAlchemy(model_class=Base)
db.init_app(app)

# Login manager setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = '請先登入以存取此頁面'

# Database Models
class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)

    # Relationship to topics
    topics = db.relationship('Topic', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        """設置密碼雜湊"""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """檢查密碼"""
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

class Topic(db.Model):
    __tablename__ = 'topics'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    emoji = db.Column(db.String(10), default='📝')
    description = db.Column(db.Text)
    date = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    # Foreign key to user
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    # Relationship to content items
    content_items = db.relationship('ContentItem', backref='topic', lazy=True, cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Topic {self.title}>'

class ContentItem(db.Model):
    __tablename__ = 'content_items'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content_type = db.Column(db.String(50), nullable=False)  # 'file', 'note', 'link'
    content = db.Column(db.Text)  # For notes or links
    created_at = db.Column(db.DateTime, default=datetime.now)

    # Foreign key to topic
    topic_id = db.Column(db.Integer, db.ForeignKey('topics.id'), nullable=False)

    def __repr__(self):
        return f'<ContentItem {self.title}>'


class FileItem(db.Model):
    __tablename__ = 'files'

    id = db.Column(db.Integer, primary_key=True)
    file_path = db.Column(db.Text, nullable=False)
    file_name = db.Column(db.String(255), nullable=False)
    original_name = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer)
    mime_type = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    topic_id = db.Column(db.Integer, nullable=False)
    # Foreign key to user
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    def __repr__(self):
        return f'<FileItem {self.file_name}>'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create tables
with app.app_context():
    db.create_all()
    logging.info("Database tables created")

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash('請輸入使用者名稱和密碼', 'error')
            return render_template('auth/login.html')

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            flash(f'歡迎回來，{user.username}！', 'success')
            return redirect(next_page or url_for('index'))
        else:
            flash('使用者名稱或密碼錯誤', 'error')

    return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        # Validation
        if not all([username, email, password, confirm_password]):
            flash('請填寫所有欄位', 'error')
            return render_template('auth/register.html')

        if password != confirm_password:
            flash('密碼確認不符', 'error')
            return render_template('auth/register.html')

        if len(password) < 6:
            flash('密碼長度至少需要6個字元', 'error')
            return render_template('auth/register.html')

        # Check if user exists
        if User.query.filter_by(username=username).first():
            flash('使用者名稱已存在', 'error')
            return render_template('auth/register.html')

        if User.query.filter_by(email=email).first():
            flash('電子郵件已被註冊', 'error')
            return render_template('auth/register.html')

        # Create new user
        try:
            user = User(username=username, email=email)
            user.set_password(password)

            db.session.add(user)
            db.session.commit()

            flash('註冊成功！請登入', 'success')
            return redirect(url_for('login'))

        except Exception as e:
            db.session.rollback()
            error_msg = str(e)
            logging.error(f"Registration error: {error_msg}")

            if 'unique constraint' in error_msg.lower():
                if 'username' in error_msg.lower():
                    flash('使用者名稱已存在', 'error')
                elif 'email' in error_msg.lower():
                    flash('電子郵件已被註冊', 'error')
                else:
                    flash('使用者名稱或電子郵件已存在', 'error')
            else:
                flash('註冊失敗，請稍後再試', 'error')

            return render_template('auth/register.html')

    return render_template('auth/register.html')

@app.route('/logout')
@login_required
def logout():
    """Logout user"""
    logout_user()
    flash('已成功登出', 'success')
    return redirect(url_for('login'))

# Main routes
@app.route('/')
@login_required
def index():
    """Main page showing all topics in grid layout"""
    topics = Topic.query.filter_by(user_id=current_user.id).order_by(Topic.created_at.desc()).all()
    return render_template('index.html', topics=topics)

@app.route('/topic/<int:topic_id>')
@login_required
def topic_detail(topic_id):
    """Topic detail page"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()
    if not topic:
        flash('主題不存在', 'error')
        return redirect(url_for('index'))

    # Get files for this user (not topic-specific, all user files)
    files = FileItem.query.filter_by(user_id=current_user.id, topic_id=topic_id).order_by(FileItem.created_at.desc()).all()

    return render_template('topic_detail.html', topic=topic, files=files)

@app.route('/add_topic', methods=['GET', 'POST'])
@login_required
def add_topic():
    """Add new topic"""
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        emoji = request.form.get('emoji', '📝').strip()
        description = request.form.get('description', '').strip()

        if not title:
            flash('請輸入主題標題', 'error')
            return redirect(url_for('index'))

        new_topic = Topic(
            title=title,
            emoji=emoji if emoji else '📝',
            description=description if description else '新增的主題',
            date=datetime.now().strftime('%Y年%m月%d日'),
            user_id=current_user.id
        )

        try:
            db.session.add(new_topic)
            db.session.commit()
            flash('主題新增成功！', 'success')
            
        except Exception as e:
            db.session.rollback()
            flash('新增失敗，請稍後再試', 'error')
            logging.error(f"Add topic error: {e}")

        return redirect(url_for('index'))

    return redirect(url_for('index'))

@app.route('/delete_topic/<int:topic_id>', methods=['POST'])
@login_required
def delete_topic(topic_id):
    """Delete a topic"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()

    if not topic:
        return jsonify({'success': False, 'error': '主題不存在'})

    try:
        db.session.delete(topic)
        db.session.commit()
        
        # 刪除 milvus 內的相關資料 (current_user.id, topic_id) 
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        logging.error(f"Delete topic error: {e}")
        return jsonify({'success': False, 'error': '刪除失敗'})

@app.route('/delete_file/<int:file_id>', methods=['POST'])
@login_required
def delete_file(file_id):
    """Delete a file"""
    topic_id = request.args.get('topic_id')
    file_item = FileItem.query.filter_by(
        id=file_id,
        user_id=current_user.id
    ).first()

    if not file_item:
        return jsonify({'success': False, 'error': '檔案不存在'})

    try:
        # Delete the physical file first
        if os.path.exists(file_item.file_path):
            try:
                os.remove(file_item.file_path)
                logging.info(f"Successfully deleted file: {file_item.file_path}")
            except Exception as file_error:
                logging.error(f"Failed to delete file {file_item.file_path}: {file_error}")
                return jsonify({'success': False, 'error': '檔案刪除失敗'})
        else:
            logging.warning(f"File not found: {file_item.file_path}")

        # Delete the FileItem from database
        db.session.delete(file_item)
        db.session.commit()
        
        delete_vector(current_user.id, topic_id, file_id)
        # 刪除 milvus 內的相關資料 (current_user.id, topic_id, file_id)
        
        

        logging.info(f"Successfully deleted file item: {file_item.original_name}")
        return jsonify({'success': True})

    except Exception as e:
        db.session.rollback()
        logging.error(f"Delete file error: {e}")
        return jsonify({'success': False, 'error': '刪除失敗'})

@app.route('/add_file/<int:topic_id>', methods=['POST'])
@login_required
def add_file(topic_id):
    """Add file to a topic - only creates FileItem record, not ContentItem"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()
    if not topic:
        flash('主題不存在', 'error')
        return redirect(url_for('index'))

    # Handle file upload
    file = request.files.get('file')
    if not file or not file.filename:
        flash('請選擇檔案', 'error')
        return redirect(url_for('topic_detail', topic_id=topic_id))

    if not allowed_file(file.filename):
        flash('不支援的檔案格式', 'error')
        return redirect(url_for('topic_detail', topic_id=topic_id))

    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        safe_filename = timestamp + filename
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
        if not os.path.exists(folder_path):
            # 如果不存在，就建立資料夾
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, safe_filename)

        # Save the file
        file.save(file_path)

        # Create FileItem record only
        file_item = FileItem(
            file_path=file_path,
            file_name=safe_filename,
            original_name=file.filename,
            file_size=os.path.getsize(file_path),
            mime_type=file.content_type,
            user_id=current_user.id,
            topic_id=topic_id
        )
        db.session.add(file_item)
        db.session.commit()

        flash('檔案上傳成功！', 'success')
        upload_file_in_milvus(int(current_user.id), int(topic_id), int(file_item.id), file_path)
        # 切分檔案，上傳到 milvus (current_user.id, topic_id, file_id, file_path)

    except Exception as e:
        db.session.rollback()
        logging.error(f"File upload error: {e}")
        flash('檔案上傳失敗，請稍後再試', 'error')

        # Clean up file if database operation failed
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

    return redirect(url_for('topic_detail', topic_id=topic_id))

@app.route('/download/<int:file_id>')
@login_required
def download_file(file_id):
    """Download a file"""
    file_item = FileItem.query.filter_by(
        id=file_id,
        user_id=current_user.id
    ).first()

    if not file_item:
        flash('檔案不存在', 'error')
        return redirect(url_for('index'))

    if not os.path.exists(file_item.file_path):
        flash('檔案已被移動或刪除', 'error')
        return redirect(url_for('index'))

    directory = os.path.dirname(file_item.file_path)
    filename = os.path.basename(file_item.file_path)

    return send_from_directory(directory, filename, as_attachment=True, download_name=file_item.original_name)

@app.route('/save_notes/<int:topic_id>', methods=['POST'])
@login_required
def save_notes(topic_id):
    """Save notes for a topic"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()
    if not topic:
        return jsonify({'success': False, 'error': '主題不存在'})

    try:
        data = request.get_json()
        notes_content = data.get('notes', '')
        
        # 更新主題的描述欄位來保存筆記
        topic.description = notes_content
        topic.updated_at = datetime.now()
        
        db.session.commit()
        return jsonify({'success': True})
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Save notes error: {e}")
        return jsonify({'success': False, 'error': '保存失敗'})

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    if current_user.is_authenticated:
        topics = Topic.query.filter_by(user_id=current_user.id).all()
        return render_template('index.html', topics=topics), 404
    else:
        return redirect(url_for('login')), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors"""
    logging.error(f"500 error: {e}")
    db.session.rollback()

    if current_user.is_authenticated:
        flash('發生內部錯誤，請稍後再試', 'error')
        return redirect(url_for('index'))
    else:
        flash('發生內部錯誤，請稍後再試', 'error')
        return redirect(url_for('login'))
    
    
@app.route('/process_selected_files', methods=['POST'])
def process_selected_files():
    data = request.get_json()
    file_ids = data.get('fileIds', [])
    print(file_ids)
    # 在這裡進行你對選中檔案的處理
    # ...
    return jsonify({'success': True})


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    file_ids = data.get('fileIds', [])
    topic_id = data.get('topicId', '')
    question = data.get('question', '')
    print(data)
    response_msg = ''

    if file_ids == []:
        response_msg = '請至少選擇一項參考來源'
        return jsonify({'success': True, 'ai_answer': response_msg})
    
    if question == '':
        response_msg = '您想問甚麼呢?'
        return jsonify({'success': True, 'ai_answer': response_msg})
    
    file_id_list = [int(file_id) for file_id in file_ids]
    
    ref_info = search_similar_embeddings(int(current_user.id), int(topic_id), file_id_list, question)

    if ref_info == []:
        response_msg = '選取檔案內沒有相關內容'
        return jsonify({'success': True, 'ai_answer': response_msg})
    
    response_msg = ask_LLM(ref_info, question)
    # 在這裡進行你對選中檔案的處理
    # ...
    return jsonify({'success': True, 'ai_answer': response_msg.replace('\n','<br>')})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from MilvusController import upload_file_in_milvus, search_similar_embeddings, delete_vector
from LLM import ask_LLM

# Set up logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
CORS(app)

# File upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'md', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Database configuration
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///notebooklm.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

db = SQLAlchemy(model_class=Base)
db.init_app(app)

# Login manager setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = '請先登入以存取此頁面'

# Database Models
class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)

    # Relationship to topics
    topics = db.relationship('Topic', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        """設置密碼雜湊"""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """檢查密碼"""
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

class Topic(db.Model):
    __tablename__ = 'topics'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    emoji = db.Column(db.String(10), default='📝')
    description = db.Column(db.Text)
    date = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    # Foreign key to user
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    # Relationship to content items
    content_items = db.relationship('ContentItem', backref='topic', lazy=True, cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Topic {self.title}>'

class ContentItem(db.Model):
    __tablename__ = 'content_items'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content_type = db.Column(db.String(50), nullable=False)  # 'file', 'note', 'link'
    content = db.Column(db.Text)  # For notes or links
    created_at = db.Column(db.DateTime, default=datetime.now)

    # Foreign key to topic
    topic_id = db.Column(db.Integer, db.ForeignKey('topics.id'), nullable=False)

    def __repr__(self):
        return f'<ContentItem {self.title}>'


class FileItem(db.Model):
    __tablename__ = 'files'

    id = db.Column(db.Integer, primary_key=True)
    file_path = db.Column(db.Text, nullable=False)
    file_name = db.Column(db.String(255), nullable=False)
    original_name = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer)
    mime_type = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    topic_id = db.Column(db.Integer, nullable=False)
    # Foreign key to user
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    def __repr__(self):
        return f'<FileItem {self.file_name}>'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create tables
with app.app_context():
    db.create_all()
    logging.info("Database tables created")

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash('請輸入使用者名稱和密碼', 'error')
            return render_template('auth/login.html')

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            flash(f'歡迎回來，{user.username}！', 'success')
            return redirect(next_page or url_for('index'))
        else:
            flash('使用者名稱或密碼錯誤', 'error')

    return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        # Validation
        if not all([username, email, password, confirm_password]):
            flash('請填寫所有欄位', 'error')
            return render_template('auth/register.html')

        if password != confirm_password:
            flash('密碼確認不符', 'error')
            return render_template('auth/register.html')

        if len(password) < 6:
            flash('密碼長度至少需要6個字元', 'error')
            return render_template('auth/register.html')

        # Check if user exists
        if User.query.filter_by(username=username).first():
            flash('使用者名稱已存在', 'error')
            return render_template('auth/register.html')

        if User.query.filter_by(email=email).first():
            flash('電子郵件已被註冊', 'error')
            return render_template('auth/register.html')

        # Create new user
        try:
            user = User(username=username, email=email)
            user.set_password(password)

            db.session.add(user)
            db.session.commit()

            flash('註冊成功！請登入', 'success')
            return redirect(url_for('login'))

        except Exception as e:
            db.session.rollback()
            error_msg = str(e)
            logging.error(f"Registration error: {error_msg}")

            if 'unique constraint' in error_msg.lower():
                if 'username' in error_msg.lower():
                    flash('使用者名稱已存在', 'error')
                elif 'email' in error_msg.lower():
                    flash('電子郵件已被註冊', 'error')
                else:
                    flash('使用者名稱或電子郵件已存在', 'error')
            else:
                flash('註冊失敗，請稍後再試', 'error')

            return render_template('auth/register.html')

    return render_template('auth/register.html')

@app.route('/logout')
@login_required
def logout():
    """Logout user"""
    logout_user()
    flash('已成功登出', 'success')
    return redirect(url_for('login'))

# Main routes
@app.route('/')
@login_required
def index():
    """Main page showing all topics in grid layout"""
    topics = Topic.query.filter_by(user_id=current_user.id).order_by(Topic.created_at.desc()).all()
    return render_template('index.html', topics=topics)

@app.route('/topic/<int:topic_id>')
@login_required
def topic_detail(topic_id):
    """Topic detail page"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()
    if not topic:
        flash('主題不存在', 'error')
        return redirect(url_for('index'))

    # Get files for this user (not topic-specific, all user files)
    files = FileItem.query.filter_by(user_id=current_user.id, topic_id=topic_id).order_by(FileItem.created_at.desc()).all()

    return render_template('topic_detail.html', topic=topic, files=files)

@app.route('/add_topic', methods=['GET', 'POST'])
@login_required
def add_topic():
    """Add new topic"""
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        emoji = request.form.get('emoji', '📝').strip()
        description = request.form.get('description', '').strip()

        if not title:
            flash('請輸入主題標題', 'error')
            return redirect(url_for('index'))

        new_topic = Topic(
            title=title,
            emoji=emoji if emoji else '📝',
            description=description if description else '新增的主題',
            date=datetime.now().strftime('%Y年%m月%d日'),
            user_id=current_user.id
        )

        try:
            db.session.add(new_topic)
            db.session.commit()
            flash('主題新增成功！', 'success')
            
        except Exception as e:
            db.session.rollback()
            flash('新增失敗，請稍後再試', 'error')
            logging.error(f"Add topic error: {e}")

        return redirect(url_for('index'))

    return redirect(url_for('index'))

@app.route('/delete_topic/<int:topic_id>', methods=['POST'])
@login_required
def delete_topic(topic_id):
    """Delete a topic"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()

    if not topic:
        return jsonify({'success': False, 'error': '主題不存在'})

    try:
        db.session.delete(topic)
        db.session.commit()
        
        # 刪除 milvus 內的相關資料 (current_user.id, topic_id) 
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        logging.error(f"Delete topic error: {e}")
        return jsonify({'success': False, 'error': '刪除失敗'})

@app.route('/delete_file/<int:file_id>', methods=['POST'])
@login_required
def delete_file(file_id):
    """Delete a file"""
    topic_id = request.args.get('topic_id')
    file_item = FileItem.query.filter_by(
        id=file_id,
        user_id=current_user.id
    ).first()

    if not file_item:
        return jsonify({'success': False, 'error': '檔案不存在'})

    try:
        # Delete the physical file first
        if os.path.exists(file_item.file_path):
            try:
                os.remove(file_item.file_path)
                logging.info(f"Successfully deleted file: {file_item.file_path}")
            except Exception as file_error:
                logging.error(f"Failed to delete file {file_item.file_path}: {file_error}")
                return jsonify({'success': False, 'error': '檔案刪除失敗'})
        else:
            logging.warning(f"File not found: {file_item.file_path}")

        # Delete the FileItem from database
        db.session.delete(file_item)
        db.session.commit()
        
        delete_vector(current_user.id, topic_id, file_id)
        # 刪除 milvus 內的相關資料 (current_user.id, topic_id, file_id)
        
        

        logging.info(f"Successfully deleted file item: {file_item.original_name}")
        return jsonify({'success': True})

    except Exception as e:
        db.session.rollback()
        logging.error(f"Delete file error: {e}")
        return jsonify({'success': False, 'error': '刪除失敗'})

@app.route('/add_file/<int:topic_id>', methods=['POST'])
@login_required
def add_file(topic_id):
    """Add file to a topic - only creates FileItem record, not ContentItem"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()
    if not topic:
        flash('主題不存在', 'error')
        return redirect(url_for('index'))

    # Handle file upload
    file = request.files.get('file')
    if not file or not file.filename:
        flash('請選擇檔案', 'error')
        return redirect(url_for('topic_detail', topic_id=topic_id))

    if not allowed_file(file.filename):
        flash('不支援的檔案格式', 'error')
        return redirect(url_for('topic_detail', topic_id=topic_id))

    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        safe_filename = timestamp + filename
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
        if not os.path.exists(folder_path):
            # 如果不存在，就建立資料夾
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, safe_filename)

        # Save the file
        file.save(file_path)

        # Create FileItem record only
        file_item = FileItem(
            file_path=file_path,
            file_name=safe_filename,
            original_name=file.filename,
            file_size=os.path.getsize(file_path),
            mime_type=file.content_type,
            user_id=current_user.id,
            topic_id=topic_id
        )
        db.session.add(file_item)
        db.session.commit()

        flash('檔案上傳成功！', 'success')
        upload_file_in_milvus(int(current_user.id), int(topic_id), int(file_item.id), file_path)
        # 切分檔案，上傳到 milvus (current_user.id, topic_id, file_id, file_path)

    except Exception as e:
        db.session.rollback()
        logging.error(f"File upload error: {e}")
        flash('檔案上傳失敗，請稍後再試', 'error')

        # Clean up file if database operation failed
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

    return redirect(url_for('topic_detail', topic_id=topic_id))

@app.route('/download/<int:file_id>')
@login_required
def download_file(file_id):
    """Download a file"""
    file_item = FileItem.query.filter_by(
        id=file_id,
        user_id=current_user.id
    ).first()

    if not file_item:
        flash('檔案不存在', 'error')
        return redirect(url_for('index'))

    if not os.path.exists(file_item.file_path):
        flash('檔案已被移動或刪除', 'error')
        return redirect(url_for('index'))

    directory = os.path.dirname(file_item.file_path)
    filename = os.path.basename(file_item.file_path)

    return send_from_directory(directory, filename, as_attachment=True, download_name=file_item.original_name)

@app.route('/save_notes/<int:topic_id>', methods=['POST'])
@login_required
def save_notes(topic_id):
    """Save notes for a topic"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()
    if not topic:
        return jsonify({'success': False, 'error': '主題不存在'})

    try:
        data = request.get_json()
        notes_content = data.get('notes', '')
        
        # 更新主題的描述欄位來保存筆記
        topic.description = notes_content
        topic.updated_at = datetime.now()
        
        db.session.commit()
        return jsonify({'success': True})
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Save notes error: {e}")
        return jsonify({'success': False, 'error': '保存失敗'})

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    if current_user.is_authenticated:
        topics = Topic.query.filter_by(user_id=current_user.id).all()
        return render_template('index.html', topics=topics), 404
    else:
        return redirect(url_for('login')), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors"""
    logging.error(f"500 error: {e}")
    db.session.rollback()

    if current_user.is_authenticated:
        flash('發生內部錯誤，請稍後再試', 'error')
        return redirect(url_for('index'))
    else:
        flash('發生內部錯誤，請稍後再試', 'error')
        return redirect(url_for('login'))
    
    
@app.route('/process_selected_files', methods=['POST'])
def process_selected_files():
    data = request.get_json()
    file_ids = data.get('fileIds', [])
    print(file_ids)
    # 在這裡進行你對選中檔案的處理
    # ...
    return jsonify({'success': True})


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    file_ids = data.get('fileIds', [])
    topic_id = data.get('topicId', '')
    question = data.get('question', '')
    print(data)
    response_msg = ''

    if file_ids == []:
        response_msg = '請至少選擇一項參考來源'
        return jsonify({'success': True, 'ai_answer': response_msg})
    
    if question == '':
        response_msg = '您想問甚麼呢?'
        return jsonify({'success': True, 'ai_answer': response_msg})
    
    file_id_list = [int(file_id) for file_id in file_ids]
    
    ref_info = search_similar_embeddings(int(current_user.id), int(topic_id), file_id_list, question)

    if ref_info == []:
        response_msg = '選取檔案內沒有相關內容'
        return jsonify({'success': True, 'ai_answer': response_msg})
    
    response_msg = ask_LLM(ref_info, question)
    # 在這裡進行你對選中檔案的處理
    # ...
    return jsonify({'success': True, 'ai_answer': response_msg.replace('\n','<br>')})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from MilvusController import upload_file_in_milvus, search_similar_embeddings, delete_vector
from LLM import ask_LLM

# Set up logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
CORS(app)

# File upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'md', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Database configuration
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///notebooklm.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

db = SQLAlchemy(model_class=Base)
db.init_app(app)

# Login manager setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = '請先登入以存取此頁面'

# Database Models
class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)

    # Relationship to topics
    topics = db.relationship('Topic', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        """設置密碼雜湊"""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """檢查密碼"""
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

class Topic(db.Model):
    __tablename__ = 'topics'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    emoji = db.Column(db.String(10), default='📝')
    description = db.Column(db.Text)
    date = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    # Foreign key to user
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    # Relationship to content items
    content_items = db.relationship('ContentItem', backref='topic', lazy=True, cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Topic {self.title}>'

class ContentItem(db.Model):
    __tablename__ = 'content_items'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content_type = db.Column(db.String(50), nullable=False)  # 'file', 'note', 'link'
    content = db.Column(db.Text)  # For notes or links
    created_at = db.Column(db.DateTime, default=datetime.now)

    # Foreign key to topic
    topic_id = db.Column(db.Integer, db.ForeignKey('topics.id'), nullable=False)

    def __repr__(self):
        return f'<ContentItem {self.title}>'


class FileItem(db.Model):
    __tablename__ = 'files'

    id = db.Column(db.Integer, primary_key=True)
    file_path = db.Column(db.Text, nullable=False)
    file_name = db.Column(db.String(255), nullable=False)
    original_name = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer)
    mime_type = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    topic_id = db.Column(db.Integer, nullable=False)
    # Foreign key to user
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    def __repr__(self):
        return f'<FileItem {self.file_name}>'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create tables
with app.app_context():
    db.create_all()
    logging.info("Database tables created")

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash('請輸入使用者名稱和密碼', 'error')
            return render_template('auth/login.html')

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            flash(f'歡迎回來，{user.username}！', 'success')
            return redirect(next_page or url_for('index'))
        else:
            flash('使用者名稱或密碼錯誤', 'error')

    return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        # Validation
        if not all([username, email, password, confirm_password]):
            flash('請填寫所有欄位', 'error')
            return render_template('auth/register.html')

        if password != confirm_password:
            flash('密碼確認不符', 'error')
            return render_template('auth/register.html')

        if len(password) < 6:
            flash('密碼長度至少需要6個字元', 'error')
            return render_template('auth/register.html')

        # Check if user exists
        if User.query.filter_by(username=username).first():
            flash('使用者名稱已存在', 'error')
            return render_template('auth/register.html')

        if User.query.filter_by(email=email).first():
            flash('電子郵件已被註冊', 'error')
            return render_template('auth/register.html')

        # Create new user
        try:
            user = User(username=username, email=email)
            user.set_password(password)

            db.session.add(user)
            db.session.commit()

            flash('註冊成功！請登入', 'success')
            return redirect(url_for('login'))

        except Exception as e:
            db.session.rollback()
            error_msg = str(e)
            logging.error(f"Registration error: {error_msg}")

            if 'unique constraint' in error_msg.lower():
                if 'username' in error_msg.lower():
                    flash('使用者名稱已存在', 'error')
                elif 'email' in error_msg.lower():
                    flash('電子郵件已被註冊', 'error')
                else:
                    flash('使用者名稱或電子郵件已存在', 'error')
            else:
                flash('註冊失敗，請稍後再試', 'error')

            return render_template('auth/register.html')

    return render_template('auth/register.html')

@app.route('/logout')
@login_required
def logout():
    """Logout user"""
    logout_user()
    flash('已成功登出', 'success')
    return redirect(url_for('login'))

# Main routes
@app.route('/')
@login_required
def index():
    """Main page showing all topics in grid layout"""
    topics = Topic.query.filter_by(user_id=current_user.id).order_by(Topic.created_at.desc()).all()
    return render_template('index.html', topics=topics)

@app.route('/topic/<int:topic_id>')
@login_required
def topic_detail(topic_id):
    """Topic detail page"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()
    if not topic:
        flash('主題不存在', 'error')
        return redirect(url_for('index'))

    # Get files for this user (not topic-specific, all user files)
    files = FileItem.query.filter_by(user_id=current_user.id, topic_id=topic_id).order_by(FileItem.created_at.desc()).all()

    return render_template('topic_detail.html', topic=topic, files=files)

@app.route('/add_topic', methods=['GET', 'POST'])
@login_required
def add_topic():
    """Add new topic"""
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        emoji = request.form.get('emoji', '📝').strip()
        description = request.form.get('description', '').strip()

        if not title:
            flash('請輸入主題標題', 'error')
            return redirect(url_for('index'))

        new_topic = Topic(
            title=title,
            emoji=emoji if emoji else '📝',
            description=description if description else '新增的主題',
            date=datetime.now().strftime('%Y年%m月%d日'),
            user_id=current_user.id
        )

        try:
            db.session.add(new_topic)
            db.session.commit()
            flash('主題新增成功！', 'success')
            
        except Exception as e:
            db.session.rollback()
            flash('新增失敗，請稍後再試', 'error')
            logging.error(f"Add topic error: {e}")

        return redirect(url_for('index'))

    return redirect(url_for('index'))

@app.route('/delete_topic/<int:topic_id>', methods=['POST'])
@login_required
def delete_topic(topic_id):
    """Delete a topic"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()

    if not topic:
        return jsonify({'success': False, 'error': '主題不存在'})

    try:
        db.session.delete(topic)
        db.session.commit()
        
        # 刪除 milvus 內的相關資料 (current_user.id, topic_id) 
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        logging.error(f"Delete topic error: {e}")
        return jsonify({'success': False, 'error': '刪除失敗'})

@app.route('/delete_file/<int:file_id>', methods=['POST'])
@login_required
def delete_file(file_id):
    """Delete a file"""
    topic_id = request.args.get('topic_id')
    file_item = FileItem.query.filter_by(
        id=file_id,
        user_id=current_user.id
    ).first()

    if not file_item:
        return jsonify({'success': False, 'error': '檔案不存在'})

    try:
        # Delete the physical file first
        if os.path.exists(file_item.file_path):
            try:
                os.remove(file_item.file_path)
                logging.info(f"Successfully deleted file: {file_item.file_path}")
            except Exception as file_error:
                logging.error(f"Failed to delete file {file_item.file_path}: {file_error}")
                return jsonify({'success': False, 'error': '檔案刪除失敗'})
        else:
            logging.warning(f"File not found: {file_item.file_path}")

        # Delete the FileItem from database
        db.session.delete(file_item)
        db.session.commit()
        
        delete_vector(current_user.id, topic_id, file_id)
        # 刪除 milvus 內的相關資料 (current_user.id, topic_id, file_id)
        
        

        logging.info(f"Successfully deleted file item: {file_item.original_name}")
        return jsonify({'success': True})

    except Exception as e:
        db.session.rollback()
        logging.error(f"Delete file error: {e}")
        return jsonify({'success': False, 'error': '刪除失敗'})

@app.route('/add_file/<int:topic_id>', methods=['POST'])
@login_required
def add_file(topic_id):
    """Add file to a topic - only creates FileItem record, not ContentItem"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()
    if not topic:
        flash('主題不存在', 'error')
        return redirect(url_for('index'))

    # Handle file upload
    file = request.files.get('file')
    if not file or not file.filename:
        flash('請選擇檔案', 'error')
        return redirect(url_for('topic_detail', topic_id=topic_id))

    if not allowed_file(file.filename):
        flash('不支援的檔案格式', 'error')
        return redirect(url_for('topic_detail', topic_id=topic_id))

    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        safe_filename = timestamp + filename
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
        if not os.path.exists(folder_path):
            # 如果不存在，就建立資料夾
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, safe_filename)

        # Save the file
        file.save(file_path)

        # Create FileItem record only
        file_item = FileItem(
            file_path=file_path,
            file_name=safe_filename,
            original_name=file.filename,
            file_size=os.path.getsize(file_path),
            mime_type=file.content_type,
            user_id=current_user.id,
            topic_id=topic_id
        )
        db.session.add(file_item)
        db.session.commit()

        flash('檔案上傳成功！', 'success')
        upload_file_in_milvus(int(current_user.id), int(topic_id), int(file_item.id), file_path)
        # 切分檔案，上傳到 milvus (current_user.id, topic_id, file_id, file_path)

    except Exception as e:
        db.session.rollback()
        logging.error(f"File upload error: {e}")
        flash('檔案上傳失敗，請稍後再試', 'error')

        # Clean up file if database operation failed
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

    return redirect(url_for('topic_detail', topic_id=topic_id))

@app.route('/download/<int:file_id>')
@login_required
def download_file(file_id):
    """Download a file"""
    file_item = FileItem.query.filter_by(
        id=file_id,
        user_id=current_user.id
    ).first()

    if not file_item:
        flash('檔案不存在', 'error')
        return redirect(url_for('index'))

    if not os.path.exists(file_item.file_path):
        flash('檔案已被移動或刪除', 'error')
        return redirect(url_for('index'))

    directory = os.path.dirname(file_item.file_path)
    filename = os.path.basename(file_item.file_path)

    return send_from_directory(directory, filename, as_attachment=True, download_name=file_item.original_name)

@app.route('/save_notes/<int:topic_id>', methods=['POST'])
@login_required
def save_notes(topic_id):
    """Save notes for a topic"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()
    if not topic:
        return jsonify({'success': False, 'error': '主題不存在'})

    try:
        data = request.get_json()
        notes_content = data.get('notes', '')
        
        # 更新主題的描述欄位來保存筆記
        topic.description = notes_content
        topic.updated_at = datetime.now()
        
        db.session.commit()
        return jsonify({'success': True})
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Save notes error: {e}")
        return jsonify({'success': False, 'error': '保存失敗'})

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    if current_user.is_authenticated:
        topics = Topic.query.filter_by(user_id=current_user.id).all()
        return render_template('index.html', topics=topics), 404
    else:
        return redirect(url_for('login')), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors"""
    logging.error(f"500 error: {e}")
    db.session.rollback()

    if current_user.is_authenticated:
        flash('發生內部錯誤，請稍後再試', 'error')
        return redirect(url_for('index'))
    else:
        flash('發生內部錯誤，請稍後再試', 'error')
        return redirect(url_for('login'))
    
    
@app.route('/process_selected_files', methods=['POST'])
def process_selected_files():
    data = request.get_json()
    file_ids = data.get('fileIds', [])
    print(file_ids)
    # 在這裡進行你對選中檔案的處理
    # ...
    return jsonify({'success': True})


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    file_ids = data.get('fileIds', [])
    topic_id = data.get('topicId', '')
    question = data.get('question', '')
    print(data)
    response_msg = ''

    if file_ids == []:
        response_msg = '請至少選擇一項參考來源'
        return jsonify({'success': True, 'ai_answer': response_msg})
    
    if question == '':
        response_msg = '您想問甚麼呢?'
        return jsonify({'success': True, 'ai_answer': response_msg})
    
    file_id_list = [int(file_id) for file_id in file_ids]
    
    ref_info = search_similar_embeddings(int(current_user.id), int(topic_id), file_id_list, question)

    if ref_info == []:
        response_msg = '選取檔案內沒有相關內容'
        return jsonify({'success': True, 'ai_answer': response_msg})
    
    response_msg = ask_LLM(ref_info, question)
    # 在這裡進行你對選中檔案的處理
    # ...
    return jsonify({'success': True, 'ai_answer': response_msg.replace('\n','<br>')})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from MilvusController import upload_file_in_milvus, search_similar_embeddings, delete_vector
from LLM import ask_LLM

# Set up logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
CORS(app)

# File upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'md', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Database configuration
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///notebooklm.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

db = SQLAlchemy(model_class=Base)
db.init_app(app)

# Login manager setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = '請先登入以存取此頁面'

# Database Models
class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)

    # Relationship to topics
    topics = db.relationship('Topic', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        """設置密碼雜湊"""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """檢查密碼"""
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

class Topic(db.Model):
    __tablename__ = 'topics'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    emoji = db.Column(db.String(10), default='📝')
    description = db.Column(db.Text)
    date = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    # Foreign key to user
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    # Relationship to content items
    content_items = db.relationship('ContentItem', backref='topic', lazy=True, cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Topic {self.title}>'

class ContentItem(db.Model):
    __tablename__ = 'content_items'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content_type = db.Column(db.String(50), nullable=False)  # 'file', 'note', 'link'
    content = db.Column(db.Text)  # For notes or links
    created_at = db.Column(db.DateTime, default=datetime.now)

    # Foreign key to topic
    topic_id = db.Column(db.Integer, db.ForeignKey('topics.id'), nullable=False)

    def __repr__(self):
        return f'<ContentItem {self.title}>'


class FileItem(db.Model):
    __tablename__ = 'files'

    id = db.Column(db.Integer, primary_key=True)
    file_path = db.Column(db.Text, nullable=False)
    file_name = db.Column(db.String(255), nullable=False)
    original_name = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer)
    mime_type = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    topic_id = db.Column(db.Integer, nullable=False)
    # Foreign key to user
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    def __repr__(self):
        return f'<FileItem {self.file_name}>'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create tables
with app.app_context():
    db.create_all()
    logging.info("Database tables created")

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash('請輸入使用者名稱和密碼', 'error')
            return render_template('auth/login.html')

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            flash(f'歡迎回來，{user.username}！', 'success')
            return redirect(next_page or url_for('index'))
        else:
            flash('使用者名稱或密碼錯誤', 'error')

    return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        # Validation
        if not all([username, email, password, confirm_password]):
            flash('請填寫所有欄位', 'error')
            return render_template('auth/register.html')

        if password != confirm_password:
            flash('密碼確認不符', 'error')
            return render_template('auth/register.html')

        if len(password) < 6:
            flash('密碼長度至少需要6個字元', 'error')
            return render_template('auth/register.html')

        # Check if user exists
        if User.query.filter_by(username=username).first():
            flash('使用者名稱已存在', 'error')
            return render_template('auth/register.html')

        if User.query.filter_by(email=email).first():
            flash('電子郵件已被註冊', 'error')
            return render_template('auth/register.html')

        # Create new user
        try:
            user = User(username=username, email=email)
            user.set_password(password)

            db.session.add(user)
            db.session.commit()

            flash('註冊成功！請登入', 'success')
            return redirect(url_for('login'))

        except Exception as e:
            db.session.rollback()
            error_msg = str(e)
            logging.error(f"Registration error: {error_msg}")

            if 'unique constraint' in error_msg.lower():
                if 'username' in error_msg.lower():
                    flash('使用者名稱已存在', 'error')
                elif 'email' in error_msg.lower():
                    flash('電子郵件已被註冊', 'error')
                else:
                    flash('使用者名稱或電子郵件已存在', 'error')
            else:
                flash('註冊失敗，請稍後再試', 'error')

            return render_template('auth/register.html')

    return render_template('auth/register.html')

@app.route('/logout')
@login_required
def logout():
    """Logout user"""
    logout_user()
    flash('已成功登出', 'success')
    return redirect(url_for('login'))

# Main routes
@app.route('/')
@login_required
def index():
    """Main page showing all topics in grid layout"""
    topics = Topic.query.filter_by(user_id=current_user.id).order_by(Topic.created_at.desc()).all()
    return render_template('index.html', topics=topics)

@app.route('/topic/<int:topic_id>')
@login_required
def topic_detail(topic_id):
    """Topic detail page"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()
    if not topic:
        flash('主題不存在', 'error')
        return redirect(url_for('index'))

    # Get files for this user (not topic-specific, all user files)
    files = FileItem.query.filter_by(user_id=current_user.id, topic_id=topic_id).order_by(FileItem.created_at.desc()).all()

    return render_template('topic_detail.html', topic=topic, files=files)

@app.route('/add_topic', methods=['GET', 'POST'])
@login_required
def add_topic():
    """Add new topic"""
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        emoji = request.form.get('emoji', '📝').strip()
        description = request.form.get('description', '').strip()

        if not title:
            flash('請輸入主題標題', 'error')
            return redirect(url_for('index'))

        new_topic = Topic(
            title=title,
            emoji=emoji if emoji else '📝',
            description=description if description else '新增的主題',
            date=datetime.now().strftime('%Y年%m月%d日'),
            user_id=current_user.id
        )

        try:
            db.session.add(new_topic)
            db.session.commit()
            flash('主題新增成功！', 'success')
            
        except Exception as e:
            db.session.rollback()
            flash('新增失敗，請稍後再試', 'error')
            logging.error(f"Add topic error: {e}")

        return redirect(url_for('index'))

    return redirect(url_for('index'))

@app.route('/delete_topic/<int:topic_id>', methods=['POST'])
@login_required
def delete_topic(topic_id):
    """Delete a topic"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()

    if not topic:
        return jsonify({'success': False, 'error': '主題不存在'})

    try:
        db.session.delete(topic)
        db.session.commit()
        
        # 刪除 milvus 內的相關資料 (current_user.id, topic_id) 
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        logging.error(f"Delete topic error: {e}")
        return jsonify({'success': False, 'error': '刪除失敗'})

@app.route('/delete_file/<int:file_id>', methods=['POST'])
@login_required
def delete_file(file_id):
    """Delete a file"""
    topic_id = request.args.get('topic_id')
    file_item = FileItem.query.filter_by(
        id=file_id,
        user_id=current_user.id
    ).first()

    if not file_item:
        return jsonify({'success': False, 'error': '檔案不存在'})

    try:
        # Delete the physical file first
        if os.path.exists(file_item.file_path):
            try:
                os.remove(file_item.file_path)
                logging.info(f"Successfully deleted file: {file_item.file_path}")
            except Exception as file_error:
                logging.error(f"Failed to delete file {file_item.file_path}: {file_error}")
                return jsonify({'success': False, 'error': '檔案刪除失敗'})
        else:
            logging.warning(f"File not found: {file_item.file_path}")

        # Delete the FileItem from database
        db.session.delete(file_item)
        db.session.commit()
        
        delete_vector(current_user.id, topic_id, file_id)
        # 刪除 milvus 內的相關資料 (current_user.id, topic_id, file_id)
        
        

        logging.info(f"Successfully deleted file item: {file_item.original_name}")
        return jsonify({'success': True})

    except Exception as e:
        db.session.rollback()
        logging.error(f"Delete file error: {e}")
        return jsonify({'success': False, 'error': '刪除失敗'})

@app.route('/add_file/<int:topic_id>', methods=['POST'])
@login_required
def add_file(topic_id):
    """Add file to a topic - only creates FileItem record, not ContentItem"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()
    if not topic:
        flash('主題不存在', 'error')
        return redirect(url_for('index'))

    # Handle file upload
    file = request.files.get('file')
    if not file or not file.filename:
        flash('請選擇檔案', 'error')
        return redirect(url_for('topic_detail', topic_id=topic_id))

    if not allowed_file(file.filename):
        flash('不支援的檔案格式', 'error')
        return redirect(url_for('topic_detail', topic_id=topic_id))

    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        safe_filename = timestamp + filename
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
        if not os.path.exists(folder_path):
            # 如果不存在，就建立資料夾
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, safe_filename)

        # Save the file
        file.save(file_path)

        # Create FileItem record only
        file_item = FileItem(
            file_path=file_path,
            file_name=safe_filename,
            original_name=file.filename,
            file_size=os.path.getsize(file_path),
            mime_type=file.content_type,
            user_id=current_user.id,
            topic_id=topic_id
        )
        db.session.add(file_item)
        db.session.commit()

        flash('檔案上傳成功！', 'success')
        upload_file_in_milvus(int(current_user.id), int(topic_id), int(file_item.id), file_path)
        # 切分檔案，上傳到 milvus (current_user.id, topic_id, file_id, file_path)

    except Exception as e:
        db.session.rollback()
        logging.error(f"File upload error: {e}")
        flash('檔案上傳失敗，請稍後再試', 'error')

        # Clean up file if database operation failed
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

    return redirect(url_for('topic_detail', topic_id=topic_id))

@app.route('/download/<int:file_id>')
@login_required
def download_file(file_id):
    """Download a file"""
    file_item = FileItem.query.filter_by(
        id=file_id,
        user_id=current_user.id
    ).first()

    if not file_item:
        flash('檔案不存在', 'error')
        return redirect(url_for('index'))

    if not os.path.exists(file_item.file_path):
        flash('檔案已被移動或刪除', 'error')
        return redirect(url_for('index'))

    directory = os.path.dirname(file_item.file_path)
    filename = os.path.basename(file_item.file_path)

    return send_from_directory(directory, filename, as_attachment=True, download_name=file_item.original_name)

@app.route('/save_notes/<int:topic_id>', methods=['POST'])
@login_required
def save_notes(topic_id):
    """Save notes for a topic"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()
    if not topic:
        return jsonify({'success': False, 'error': '主題不存在'})

    try:
        data = request.get_json()
        notes_content = data.get('notes', '')
        
        # 更新主題的描述欄位來保存筆記
        topic.description = notes_content
        topic.updated_at = datetime.now()
        
        db.session.commit()
        return jsonify({'success': True})
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Save notes error: {e}")
        return jsonify({'success': False, 'error': '保存失敗'})

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    if current_user.is_authenticated:
        topics = Topic.query.filter_by(user_id=current_user.id).all()
        return render_template('index.html', topics=topics), 404
    else:
        return redirect(url_for('login')), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors"""
    logging.error(f"500 error: {e}")
    db.session.rollback()

    if current_user.is_authenticated:
        flash('發生內部錯誤，請稍後再試', 'error')
        return redirect(url_for('index'))
    else:
        flash('發生內部錯誤，請稍後再試', 'error')
        return redirect(url_for('login'))
    
    
@app.route('/process_selected_files', methods=['POST'])
def process_selected_files():
    data = request.get_json()
    file_ids = data.get('fileIds', [])
    print(file_ids)
    # 在這裡進行你對選中檔案的處理
    # ...
    return jsonify({'success': True})


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    file_ids = data.get('fileIds', [])
    topic_id = data.get('topicId', '')
    question = data.get('question', '')
    print(data)
    response_msg = ''

    if file_ids == []:
        response_msg = '請至少選擇一項參考來源'
        return jsonify({'success': True, 'ai_answer': response_msg})
    
    if question == '':
        response_msg = '您想問甚麼呢?'
        return jsonify({'success': True, 'ai_answer': response_msg})
    
    file_id_list = [int(file_id) for file_id in file_ids]
    
    ref_info = search_similar_embeddings(int(current_user.id), int(topic_id), file_id_list, question)

    if ref_info == []:
        response_msg = '選取檔案內沒有相關內容'
        return jsonify({'success': True, 'ai_answer': response_msg})
    
    response_msg = ask_LLM(ref_info, question)
    # 在這裡進行你對選中檔案的處理
    # ...
    return jsonify({'success': True, 'ai_answer': response_msg.replace('\n','<br>')})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from MilvusController import upload_file_in_milvus, search_similar_embeddings, delete_vector
from LLM import ask_LLM

# Set up logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
CORS(app)

# File upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'md', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Database configuration
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///notebooklm.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

db = SQLAlchemy(model_class=Base)
db.init_app(app)

# Login manager setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = '請先登入以存取此頁面'

# Database Models
class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)

    # Relationship to topics
    topics = db.relationship('Topic', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        """設置密碼雜湊"""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """檢查密碼"""
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

class Topic(db.Model):
    __tablename__ = 'topics'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    emoji = db.Column(db.String(10), default='📝')
    description = db.Column(db.Text)
    date = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    # Foreign key to user
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    # Relationship to content items
    content_items = db.relationship('ContentItem', backref='topic', lazy=True, cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Topic {self.title}>'

class ContentItem(db.Model):
    __tablename__ = 'content_items'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content_type = db.Column(db.String(50), nullable=False)  # 'file', 'note', 'link'
    content = db.Column(db.Text)  # For notes or links
    created_at = db.Column(db.DateTime, default=datetime.now)

    # Foreign key to topic
    topic_id = db.Column(db.Integer, db.ForeignKey('topics.id'), nullable=False)

    def __repr__(self):
        return f'<ContentItem {self.title}>'


class FileItem(db.Model):
    __tablename__ = 'files'

    id = db.Column(db.Integer, primary_key=True)
    file_path = db.Column(db.Text, nullable=False)
    file_name = db.Column(db.String(255), nullable=False)
    original_name = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer)
    mime_type = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    topic_id = db.Column(db.Integer, nullable=False)
    # Foreign key to user
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    def __repr__(self):
        return f'<FileItem {self.file_name}>'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create tables
with app.app_context():
    db.create_all()
    logging.info("Database tables created")

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash('請輸入使用者名稱和密碼', 'error')
            return render_template('auth/login.html')

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            flash(f'歡迎回來，{user.username}！', 'success')
            return redirect(next_page or url_for('index'))
        else:
            flash('使用者名稱或密碼錯誤', 'error')

    return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        # Validation
        if not all([username, email, password, confirm_password]):
            flash('請填寫所有欄位', 'error')
            return render_template('auth/register.html')

        if password != confirm_password:
            flash('密碼確認不符', 'error')
            return render_template('auth/register.html')

        if len(password) < 6:
            flash('密碼長度至少需要6個字元', 'error')
            return render_template('auth/register.html')

        # Check if user exists
        if User.query.filter_by(username=username).first():
            flash('使用者名稱已存在', 'error')
            return render_template('auth/register.html')

        if User.query.filter_by(email=email).first():
            flash('電子郵件已被註冊', 'error')
            return render_template('auth/register.html')

        # Create new user
        try:
            user = User(username=username, email=email)
            user.set_password(password)

            db.session.add(user)
            db.session.commit()

            flash('註冊成功！請登入', 'success')
            return redirect(url_for('login'))

        except Exception as e:
            db.session.rollback()
            error_msg = str(e)
            logging.error(f"Registration error: {error_msg}")

            if 'unique constraint' in error_msg.lower():
                if 'username' in error_msg.lower():
                    flash('使用者名稱已存在', 'error')
                elif 'email' in error_msg.lower():
                    flash('電子郵件已被註冊', 'error')
                else:
                    flash('使用者名稱或電子郵件已存在', 'error')
            else:
                flash('註冊失敗，請稍後再試', 'error')

            return render_template('auth/register.html')

    return render_template('auth/register.html')

@app.route('/logout')
@login_required
def logout():
    """Logout user"""
    logout_user()
    flash('已成功登出', 'success')
    return redirect(url_for('login'))

# Main routes
@app.route('/')
@login_required
def index():
    """Main page showing all topics in grid layout"""
    topics = Topic.query.filter_by(user_id=current_user.id).order_by(Topic.created_at.desc()).all()
    return render_template('index.html', topics=topics)

@app.route('/topic/<int:topic_id>')
@login_required
def topic_detail(topic_id):
    """Topic detail page"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()
    if not topic:
        flash('主題不存在', 'error')
        return redirect(url_for('index'))

    # Get files for this user (not topic-specific, all user files)
    files = FileItem.query.filter_by(user_id=current_user.id, topic_id=topic_id).order_by(FileItem.created_at.desc()).all()

    return render_template('topic_detail.html', topic=topic, files=files)

@app.route('/add_topic', methods=['GET', 'POST'])
@login_required
def add_topic():
    """Add new topic"""
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        emoji = request.form.get('emoji', '📝').strip()
        description = request.form.get('description', '').strip()

        if not title:
            flash('請輸入主題標題', 'error')
            return redirect(url_for('index'))

        new_topic = Topic(
            title=title,
            emoji=emoji if emoji else '📝',
            description=description if description else '新增的主題',
            date=datetime.now().strftime('%Y年%m月%d日'),
            user_id=current_user.id
        )

        try:
            db.session.add(new_topic)
            db.session.commit()
            flash('主題新增成功！', 'success')
            
        except Exception as e:
            db.session.rollback()
            flash('新增失敗，請稍後再試', 'error')
            logging.error(f"Add topic error: {e}")

        return redirect(url_for('index'))

    return redirect(url_for('index'))

@app.route('/delete_topic/<int:topic_id>', methods=['POST'])
@login_required
def delete_topic(topic_id):
    """Delete a topic"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()

    if not topic:
        return jsonify({'success': False, 'error': '主題不存在'})

    try:
        db.session.delete(topic)
        db.session.commit()
        
        # 刪除 milvus 內的相關資料 (current_user.id, topic_id) 
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        logging.error(f"Delete topic error: {e}")
        return jsonify({'success': False, 'error': '刪除失敗'})

@app.route('/delete_file/<int:file_id>', methods=['POST'])
@login_required
def delete_file(file_id):
    """Delete a file"""
    topic_id = request.args.get('topic_id')
    file_item = FileItem.query.filter_by(
        id=file_id,
        user_id=current_user.id
    ).first()

    if not file_item:
        return jsonify({'success': False, 'error': '檔案不存在'})

    try:
        # Delete the physical file first
        if os.path.exists(file_item.file_path):
            try:
                os.remove(file_item.file_path)
                logging.info(f"Successfully deleted file: {file_item.file_path}")
            except Exception as file_error:
                logging.error(f"Failed to delete file {file_item.file_path}: {file_error}")
                return jsonify({'success': False, 'error': '檔案刪除失敗'})
        else:
            logging.warning(f"File not found: {file_item.file_path}")

        # Delete the FileItem from database
        db.session.delete(file_item)
        db.session.commit()
        
        delete_vector(current_user.id, topic_id, file_id)
        # 刪除 milvus 內的相關資料 (current_user.id, topic_id, file_id)
        
        

        logging.info(f"Successfully deleted file item: {file_item.original_name}")
        return jsonify({'success': True})

    except Exception as e:
        db.session.rollback()
        logging.error(f"Delete file error: {e}")
        return jsonify({'success': False, 'error': '刪除失敗'})

@app.route('/add_file/<int:topic_id>', methods=['POST'])
@login_required
def add_file(topic_id):
    """Add file to a topic - only creates FileItem record, not ContentItem"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()
    if not topic:
        flash('主題不存在', 'error')
        return redirect(url_for('index'))

    # Handle file upload
    file = request.files.get('file')
    if not file or not file.filename:
        flash('請選擇檔案', 'error')
        return redirect(url_for('topic_detail', topic_id=topic_id))

    if not allowed_file(file.filename):
        flash('不支援的檔案格式', 'error')
        return redirect(url_for('topic_detail', topic_id=topic_id))

    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        safe_filename = timestamp + filename
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
        if not os.path.exists(folder_path):
            # 如果不存在，就建立資料夾
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, safe_filename)

        # Save the file
        file.save(file_path)

        # Create FileItem record only
        file_item = FileItem(
            file_path=file_path,
            file_name=safe_filename,
            original_name=file.filename,
            file_size=os.path.getsize(file_path),
            mime_type=file.content_type,
            user_id=current_user.id,
            topic_id=topic_id
        )
        db.session.add(file_item)
        db.session.commit()

        flash('檔案上傳成功！', 'success')
        upload_file_in_milvus(int(current_user.id), int(topic_id), int(file_item.id), file_path)
        # 切分檔案，上傳到 milvus (current_user.id, topic_id, file_id, file_path)

    except Exception as e:
        db.session.rollback()
        logging.error(f"File upload error: {e}")
        flash('檔案上傳失敗，請稍後再試', 'error')

        # Clean up file if database operation failed
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

    return redirect(url_for('topic_detail', topic_id=topic_id))

@app.route('/download/<int:file_id>')
@login_required
def download_file(file_id):
    """Download a file"""
    file_item = FileItem.query.filter_by(
        id=file_id,
        user_id=current_user.id
    ).first()

    if not file_item:
        flash('檔案不存在', 'error')
        return redirect(url_for('index'))

    if not os.path.exists(file_item.file_path):
        flash('檔案已被移動或刪除', 'error')
        return redirect(url_for('index'))

    directory = os.path.dirname(file_item.file_path)
    filename = os.path.basename(file_item.file_path)

    return send_from_directory(directory, filename, as_attachment=True, download_name=file_item.original_name)

@app.route('/save_notes/<int:topic_id>', methods=['POST'])
@login_required
def save_notes(topic_id):
    """Save notes for a topic"""
    topic = Topic.query.filter_by(id=topic_id, user_id=current_user.id).first()
    if not topic:
        return jsonify({'success': False, 'error': '主題不存在'})

    try:
        data = request.get_json()
        notes_content = data.get('notes', '')
        
        # 更新主題的描述欄位來保存筆記
        topic.description = notes_content
        topic.updated_at = datetime.now()
        
        db.session.commit()
        return jsonify({'success': True})
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Save notes error: {e}")
        return jsonify({'success': False, 'error': '保存失敗'})

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    if current_user.is_authenticated:
        topics = Topic.query.filter_by(user_id=current_user.id).all()
        return render_template('index.html', topics=topics), 404
    else:
        return redirect(url_for('login')), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors"""
    logging.error(f"500 error: {e}")
    db.session.rollback()

    if current_user.is_authenticated:
        flash('發生內部錯誤，請稍後再試', 'error')
        return redirect(url_for('index'))
    else:
        flash('發生內部錯誤，請稍後再試', 'error')
        return redirect(url_for('login'))
    
    
@app.route('/process_selected_files', methods=['POST'])
def process_selected_files():
    data = request.get_json()
    file_ids = data.get('fileIds', [])
    print(file_ids)
    # 在這裡進行你對選中檔案的處理
    # ...
    return jsonify({'success': True})


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    file_ids = data.get('fileIds', [])
    topic_id = data.get('topicId', '')
    question = data.get('question', '')
    print(data)
    response_msg = ''

    if file_ids == []:
        response_msg = '請至少選擇一項參考來源'
        return jsonify({'success': True, 'ai_answer': response_msg})
    
    if question == '':
        response_msg = '您想問甚麼呢?'
        return jsonify({'success': True, 'ai_answer': response_msg})
    
    file_id_list = [int(file_id) for file_id in file_ids]
    
    ref_info = search_similar_embeddings(int(current_user.id), int(topic_id), file_id_list, question)

    if ref_info == []:
        response_msg = '選取檔案內沒有相關內容'
        return jsonify({'success': True, 'ai_answer': response_msg})
    
    response_msg = ask_LLM(ref_info, question)
    # 在這裡進行你對選中檔案的處理
    # ...
    return jsonify({'success': True, 'ai_answer': response_msg.replace('\n','<br>')})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)