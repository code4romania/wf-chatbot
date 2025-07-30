<template>
  <div class="container">
    <h1 class="text-center">Dopomoha Chat Interface</h1>

    <div class="chat-area">
      <div class="messages-display">
        <div v-for="(message, index) in chatMessages" :key="index" :class="['message-bubble', message.type]">
          <p v-html="message.text"></p>
          <div v-if="message.type === 'bot' && message.answerData" class="bot-message-actions">
            <AnswerCard
              :answer="message.answerData"
              :session-id="sessionId"
              :current-k-position="1"
              :review-disabled="isReviewDisabledForCurrentK"
              @reviewSubmitted="handleReviewSubmitted"
              :query-id="message.queryId"
            />
          </div>
        </div>
      </div>

      <div class="query-section">
        <input type="text" v-model="userQuery" placeholder="Type your message..." @keyup.enter="sendQuery" :disabled="loading" />
        <button @click="sendQuery" :disabled="loading">Send</button>
      </div>
    </div>

    <div v-if="loading" class="info-message loading-message">Typing...</div>
    <div v-if="queryError" class="error-message">{{ queryError }}</div>
  </div>
</template>

<script setup>
import { ref, computed, nextTick } from 'vue';

// Reactive state variables
const userQuery = ref('');
const chatMessages = ref([]); // Stores objects like { type: 'user' | 'bot', text: 'message text', answerData: {}, queryId: '' }
const loading = ref(false);
const queryError = ref(null);
const sessionId = ref(null); // Session ID for continuity, if desired by the backend
const reviewedKPositions = ref(new Set()); // Stores 1-indexed k positions that have been reviewed (always 1 for chat)

// FastAPI Backend URL
const API_BASE_URL = 'http://localhost:8000';

// --- Computed property for review constraint ---
const isReviewDisabledForCurrentK = computed(() => {
  // In a k=1 chat, review is always enabled if an answer is present,
  // unless we want to prevent multiple reviews for the same query.
  // For simplicity, we'll allow review if an answer is displayed.
  return false; // Always allow review for k=1 in chat context
});

// --- Chat Message Management ---
const addMessage = (type, text, answerData = null, queryId = null) => {
  chatMessages.value.push({ type, text, answerData, queryId });
  scrollToBottom();
};

const scrollToBottom = async () => {
  await nextTick(); // Ensure DOM is updated before scrolling
  const messagesDisplay = document.querySelector('.messages-display');
  if (messagesDisplay) {
    messagesDisplay.scrollTop = messagesDisplay.scrollHeight;
  }
};

// --- Query Submission Function ---
const sendQuery = async () => {
  if (!userQuery.value.trim()) {
    queryError.value = "Please enter a message.";
    return;
  }

  const queryToSend = userQuery.value;
  addMessage('user', queryToSend); // Add user message to chat

  loading.value = true;
  queryError.value = null;
  userQuery.value = ''; // Clear input field immediately
  reviewedKPositions.value.clear(); // Reset reviewed positions for new query

  try {
    const response = await fetch(`${API_BASE_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: queryToSend,
        top_k: 1, // Always request only the top 1 answer for chat
        metric: 'cosine',
        session_id: sessionId.value,
        use_concat_matcher: false, // You can make this dynamic if needed
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to get a response.');
    }

    const data = await response.json();
    console.log(data)
    const botAnswer = data.results[0]; // Get the single top answer
    sessionId.value = data.session_id;
    const currentQueryId = data.query_id;

    if (botAnswer && botAnswer.response) {
      addMessage('bot', botAnswer.answer, botAnswer, currentQueryId);
    } else {
      addMessage('bot', "I'm sorry, I couldn't find a relevant answer for that.");
    }

  } catch (err) {
    queryError.value = err.message;
    console.error('Chat query error:', err);
    addMessage('bot', "Oops! Something went wrong. Please try again.");
  } finally {
    loading.value = false;
  }
};

// --- Event handler for review submission from AnswerCard ---
const handleReviewSubmitted = (reviewData) => {
  console.log('Review submitted from component:', reviewData);
  // For k=1 chat, we just acknowledge it.
  // If you want to prevent multiple reviews for the *same* specific bot message,
  // you could store the queryId here.
};
</script>

<style scoped>
/* Scoped styles ensure they only apply to this component */
.container {
  max-width: 700px;
  margin: 40px auto;
  padding: 30px;
  background-color: #ffffff;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  height: 80vh; /* Make container fill most of the viewport height */
  min-height: 500px;
}

.text-center {
  text-align: center;
  margin-bottom: 20px;
  color: #2c3e50;
}

.chat-area {
  flex-grow: 1; /* Allows chat area to expand and take available space */
  display: flex;
  flex-direction: column;
  border: 1px solid #e0e0e0;
  border-radius: 10px;
  overflow: hidden; /* Important for containing messages-display scroll */
  background-color: #fcfcfc;
}

.messages-display {
  flex-grow: 1; /* Messages area takes available space */
  padding: 20px;
  overflow-y: auto; /* Enable scrolling for messages */
  display: flex;
  flex-direction: column; /* Messages stack vertically */
  gap: 15px; /* Space between message bubbles */
}

.message-bubble {
  max-width: 80%;
  padding: 12px 18px;
  border-radius: 20px;
  line-height: 1.5;
  word-wrap: break-word; /* Ensure long words wrap */
}

.message-bubble.user {
  align-self: flex-end; /* User messages on the right */
  background-color: #007bff;
  color: white;
  border-bottom-right-radius: 4px; /* Slightly different corner for visual distinction */
}

.message-bubble.bot {
  align-self: flex-start; /* Bot messages on the left */
  background-color: #e0e0e0;
  color: #333;
  border-bottom-left-radius: 4px; /* Slightly different corner */
}

.bot-message-actions {
  margin-top: 10px; /* Space between bot message text and review card */
}

.query-section {
  display: flex;
  gap: 10px;
  padding: 15px;
  border-top: 1px solid #e0e0e0; /* Separator from messages */
  background-color: #f0f2f5; /* Background for input area */
}

.query-section input[type="text"] {
  flex-grow: 1;
  padding: 12px 18px;
  border: 1px solid #dcdcdc;
  border-radius: 25px; /* Pill-shaped input */
  font-size: 1rem;
  transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

.query-section input[type="text"]:focus {
  border-color: #007bff;
  box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.25);
  outline: none;
}

.query-section button {
  padding: 12px 25px;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 25px; /* Pill-shaped button */
  cursor: pointer;
  font-size: 1rem;
  font-weight: bold;
  transition: background-color 0.3s ease, transform 0.2s ease;
}

.query-section button:hover:not(:disabled) {
  background-color: #0056b3;
  transform: translateY(-1px);
}

.query-section button:disabled {
  background-color: #cccccc;
  cursor: not-allowed;
  opacity: 0.7;
}

/* Info and Error Messages */
.info-message {
  padding: 10px;
  margin-top: 15px;
  border-radius: 8px;
  font-size: 0.9rem;
  text-align: center;
}

.loading-message {
  background-color: #e0f7fa;
  color: #007bff;
  border: 1px solid #b2ebf2;
}

.error-message {
  background-color: #ffe0e0;
  color: #d32f2f;
  border: 1px solid #ef9a9a;
  padding: 10px;
  margin-top: 15px;
  border-radius: 8px;
  font-size: 0.9rem;
  text-align: center;
}
</style>