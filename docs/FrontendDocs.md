# Dopomoha Frontend


- **Smart FAQ**: lets users input a query and browse multiple question-answer pairs.
- **Chat Interface**: provides a conversational interface with one answer per message.

## Table of Contents

- [Setup](#setup)  
- [Pages Overview](#pages-overview)  
  - [1. Dopomoha Smart FAQ](#1-dopomoha-smart-faq)  
  - [2. Dopomoha Chat Interface](#2-dopomoha-chat-interface)  


---

## Setup

1. Go to the frontend folder:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

The app will be available at: [http://localhost:3000](http://localhost:3000)

**or simply run the docker compose up from root directory where docker-compose.yml lives**

---

## Pages Overview

### 1. Dopomoha Smart FAQ

- Input a query and receive **top-k answers** (default: 3).
- Navigate through answers using **Previous/Next** buttons.
- You can only review answers after reviewing the first one.
- Backend call: `POST /query` with `top_k = 3`.

### 2. Dopomoha Chat Interface

- Chat UI for interacting with the system like a messaging app.
- Each message gets **top-1 answer** only.
- Bot replies may include reviewable answers.
- Backend call: `POST /query` with `top_k = 1`.
