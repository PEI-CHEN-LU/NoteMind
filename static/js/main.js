// Main JavaScript file for NotebookLM Plus

document.addEventListener('DOMContentLoaded', function() {
    // Initialize delete functionality
    initializeDeleteFunctionality();
    
    // Initialize form validation
    initializeFormValidation();
    
    // Initialize tooltips
    initializeTooltips();
    
    // Initialize card animations
    initializeCardAnimations();
});

/**
 * Initialize delete functionality with confirmation
 */
function initializeDeleteFunctionality() {
    let topicToDelete = null;
    const deleteModal = document.getElementById('deleteTopicModal');
    
    // Handle delete button clicks using event delegation
    document.addEventListener('click', function(e) {
        // Check if the clicked element or its parent has the delete-topic class
        let deleteBtn = null;
        if (e.target.classList.contains('delete-topic')) {
            deleteBtn = e.target;
        } else if (e.target.closest('.delete-topic')) {
            deleteBtn = e.target.closest('.delete-topic');
        }
        
        if (deleteBtn) {
            console.log('刪除按鈕被點擊');
            e.preventDefault();
            e.stopPropagation();
            
            topicToDelete = deleteBtn.getAttribute('data-topic-id');
            console.log('要刪除的主題 ID:', topicToDelete);
            
            if (deleteModal) {
                const modal = new bootstrap.Modal(deleteModal);
                modal.show();
            }
        }
    });
    
    // Handle confirm delete
    const confirmDeleteBtn = document.getElementById('confirmDelete');
    if (confirmDeleteBtn) {
        confirmDeleteBtn.addEventListener('click', function() {
            console.log('確認刪除按鈕被點擊, 主題ID:', topicToDelete);
            if (topicToDelete) {
                deleteTopic(topicToDelete);
            }
        });
    }
}

/**
 * Delete a topic via AJAX
 */
function deleteTopic(topicId) {
    console.log('刪除主題 ID:', topicId);
    const deleteUrl = `/delete_topic/${topicId}`;
    
    // 添加 loading 狀態
    const confirmBtn = document.getElementById('confirmDelete');
    if (confirmBtn) {
        confirmBtn.disabled = true;
        confirmBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>刪除中...';
    }
    
    fetch(deleteUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => {
        console.log('Response status:', response.status);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Response data:', data);
        if (data.success) {
            // Close modal first
            const deleteModal = document.getElementById('deleteTopicModal');
            if (deleteModal) {
                const modal = bootstrap.Modal.getInstance(deleteModal);
                if (modal) {
                    modal.hide();
                }
            }
            
            // Remove the topic card from the DOM
            const topicCard = document.querySelector(`[data-topic-id="${topicId}"]`);
            if (topicCard) {
                topicCard.style.animation = 'fadeOut 0.3s ease-out';
                setTimeout(() => {
                    topicCard.remove();
                    checkEmptyState();
                }, 300);
            }
            
            // Show success message
            showAlert('主題已成功刪除', 'success');
            
            // If we're on the detail page, redirect to home
            if (window.location.pathname.includes('/topic/')) {
                setTimeout(() => {
                    window.location.href = '/';
                }, 1000);
            }
        } else {
            showAlert('刪除失敗，請稍後再試', 'error');
        }
    })
    .catch(error => {
        console.error('刪除錯誤:', error);
        showAlert('刪除失敗，請稍後再試', 'error');
    })
    .finally(() => {
        // 重置 loading 狀態
        if (confirmBtn) {
            confirmBtn.disabled = false;
            confirmBtn.innerHTML = '確認刪除';
        }
    });
}

/**
 * Initialize form validation
 */
function initializeFormValidation() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const titleInput = form.querySelector('input[name="title"]');
            
            if (titleInput && !titleInput.value.trim()) {
                e.preventDefault();
                showAlert('請輸入主題標題', 'error');
                titleInput.focus();
            }
        });
    });
    
    // Real-time validation for title input
    const titleInput = document.querySelector('input[name="title"]');
    if (titleInput) {
        titleInput.addEventListener('input', function() {
            const submitBtn = this.form.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = !this.value.trim();
            }
        });
    }
}

/**
 * Initialize tooltips
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Initialize card animations
 */
function initializeCardAnimations() {
    const cards = document.querySelectorAll('.topic-card');
    
    cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
    });
    
    // Add hover effect enhancement
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-4px)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
}

/**
 * Show alert message
 */
function showAlert(message, type) {
    const alertContainer = document.createElement('div');
    alertContainer.className = `alert alert-${type === 'error' ? 'danger' : 'success'} alert-dismissible fade show position-fixed`;
    alertContainer.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    alertContainer.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertContainer);
    
    // Auto-dismiss after 3 seconds
    setTimeout(() => {
        if (alertContainer.parentNode) {
            alertContainer.remove();
        }
    }, 3000);
}

/**
 * Check if we need to show empty state
 */
function checkEmptyState() {
    const topicsGrid = document.querySelector('.topics-grid');
    if (!topicsGrid) return;
    
    const topicCards = topicsGrid.querySelectorAll('.topic-card');
    const existingEmptyState = topicsGrid.querySelector('.empty-state');
    
    if (topicCards.length === 0 && !existingEmptyState) {
        const emptyState = document.createElement('div');
        emptyState.className = 'empty-state';
        emptyState.innerHTML = `
            <div class="empty-icon">📝</div>
            <h3>還沒有任何主題</h3>
            <p>點擊「新增」按鈕來建立你的第一個主題</p>
        `;
        topicsGrid.appendChild(emptyState);
    }
}

/**
 * Add fade out animation to CSS
 */
const fadeOutCSS = `
@keyframes fadeOut {
    from {
        opacity: 1;
        transform: translateY(0);
    }
    to {
        opacity: 0;
        transform: translateY(-20px);
    }
}
`;

// Add the CSS to the page
const style = document.createElement('style');
style.textContent = fadeOutCSS;
document.head.appendChild(style);

// Handle emoji picker for better UX
document.addEventListener('DOMContentLoaded', function() {
    const emojiInput = document.querySelector('input[name="emoji"]');
    if (emojiInput) {
        // Common emojis for topics
        const commonEmojis = ['📝', '💡', '🔬', '📊', '🎯', '🚀', '💻', '📚', '🌟', '🎨', '🔧', '📱'];
        
        emojiInput.addEventListener('focus', function() {
            if (!this.value) {
                this.placeholder = commonEmojis[Math.floor(Math.random() * commonEmojis.length)];
            }
        });
    }
});
