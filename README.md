# Galileo Project

This is the monorepo for the Galileo project, containing both the backend and frontend applications.

## Project Structure

- **`galileo/`**: Backend application (FastAPI, Python)
- **`galileo_frontend/`**: Frontend application (React, TypeScript)

## Getting Started

### Backend (`galileo`)

1. Navigate to the backend directory:
   ```bash
   cd galileo
   ```
2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
3. Start the server:
   ```bash
   uvicorn main:app --reload
   ```
   The backend will run at `http://localhost:8000` (or similar port).

### Frontend (`galileo_frontend`)

1. Navigate to the frontend directory:
   ```bash
   cd galileo_frontend
   ```
2. Install dependencies (if not already installed):
   ```bash
   pnpm install
   ```
3. Start the development server:
   ```bash
   pnpm start
   ```
   The frontend will run at `http://localhost:3000`.

## Git Structure

This repository is a **monorepo**.
- The root directory contains the git configuration.
- Previous separate repositories for backend and frontend have been merged into this single repository.
- `galileo/.git` and `galileo_frontend/.git` were archived to `.git_backup` to avoid conflicts.
