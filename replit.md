# NotebookLM Plus

## Overview

NotebookLM Plus is a Flask-based web application that provides a simplified interface for managing topics and notes, inspired by Google's NotebookLM. The application serves as a topic management system with a clean, Bootstrap-powered frontend that supports both Chinese and English content.

## System Architecture

The application follows a MVC (Model-View-Controller) pattern with Flask as the web framework:

- **Frontend**: Bootstrap 5 + custom CSS for responsive UI
- **Backend**: Flask with PostgreSQL database and Flask-Login authentication
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Authentication**: Flask-Login with password hashing
- **Template Engine**: Jinja2 for server-side rendering
- **Static Assets**: CSS, JavaScript, and Font Awesome icons

## Key Components

### Backend (`app.py` & `models.py`)
- **Flask Application**: Main application instance with CORS enabled
- **Database Models**: User and Topic models with SQLAlchemy ORM
- **Authentication**: Flask-Login for user session management
- **Session Management**: Flask's built-in session handling with secret key
- **Logging**: Debug-level logging for development

### Frontend Templates
- **Base Template**: Common layout with navigation, Bootstrap, and Font Awesome
- **Index Template**: Main dashboard displaying topics in a grid layout
- **Topic Detail Template**: Individual topic view with breadcrumbs and actions

### Static Assets
- **CSS**: Custom styling built on top of Bootstrap 5
- **JavaScript**: Client-side functionality for delete operations, form validation, and UI interactions

### Database Schema
**User Model:**
- id (Primary Key)
- username (Unique)
- email (Unique)
- password_hash (Hashed password)
- created_at (Timestamp)

**Topic Model:**
- id (Primary Key)
- title (Topic title)
- emoji (Display emoji)
- description (Topic description)
- date (Formatted date string)
- user_id (Foreign Key to User)
- created_at/updated_at (Timestamps)

## Data Flow

1. **Authentication**: User login/registration through Flask-Login
2. **Request Handling**: Flask routes handle HTTP requests with authentication checks
3. **Database Operations**: SQLAlchemy ORM handles database queries and user-specific data
4. **Template Rendering**: Jinja2 templates render HTML with authenticated user data
5. **Client Interaction**: JavaScript handles user interactions (delete, form validation)
6. **Response**: HTML pages or JSON responses sent to client

## External Dependencies

### Python Packages
- **Flask**: Web framework for handling HTTP requests and responses
- **Flask-CORS**: Cross-Origin Resource Sharing support
- **Flask-SQLAlchemy**: Database ORM for PostgreSQL
- **Flask-Login**: User session management and authentication
- **Werkzeug**: Password hashing utilities

### Frontend Libraries
- **Bootstrap 5.3.0**: CSS framework for responsive design
- **Font Awesome 6.4.0**: Icon library for UI elements

## Deployment Strategy

The application is configured for development deployment:
- **Host**: `0.0.0.0` (accessible from all interfaces)
- **Port**: `5000`
- **Debug Mode**: Enabled for development
- **Entry Point**: `main.py` serves as the application entry point

**Note**: The application now uses PostgreSQL database for persistent data storage. User data and topics are preserved between application restarts.

## User Preferences

Preferred communication style: Simple, everyday language.

## Changelog

Changelog:
- July 07, 2025. Initial setup