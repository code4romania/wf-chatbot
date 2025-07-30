<template>
  <div class="container">
    <h1 class="text-center">Dopomoha Smart FAQ</h1>

    <div class="query-section">
      <input type="text" v-model="userQuery" placeholder="Enter your query..." @keyup.enter="sendQuery" />
      <button @click="sendQuery" :disabled="loading">Search</button>
    </div>

    <div v-if="loading" class="info-message loading-message">Loading answers...</div>
    <div v-if="queryError" class="error-message">{{ queryError }}</div>

    <div v-if="answers.length > 0" class="answers-carousel">
      <div class="carousel-navigation">
        <button @click="prevAnswer" :disabled="currentAnswerIndex === 0">← Previous</button>
        <span class="carousel-counter">{{ currentAnswerIndex + 1 }} / {{ answers.length }}</span>
        <button @click="nextAnswer" :disabled="currentAnswerIndex === answers.length - 1">Next →</button>
      </div>

      <AnswerCard
        v-if="currentAnswer"
        :debug=true
        :answer="currentAnswer"
        :session-id="sessionId"
        :current-k-position="currentAnswerIndex + 1"
        :review-disabled="isReviewDisabledForCurrentK"
        @reviewSubmitted="handleReviewSubmitted"
        :query-id="queryId"
      />
    </div>

    <div v-else-if="!loading && !queryError && !answers.length">
      <p class="text-center no-results-message">Submit a query to see answers.</p>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'; // Import onMounted

// Reactive state variables
const userQuery = ref('');
const topK = ref(3);
const answers = ref([]);
const loading = ref(false);
const queryError = ref(null);
const currentAnswerIndex = ref(0);
const sessionId = ref(null);
const queryId = ref(null);
const useConcatMatcher = ref(false);

const reviewedKPositions = ref(new Set()); // Stores 1-indexed k positions that have been reviewed for the current session/query

// FastAPI Backend URL
const API_BASE_URL = 'http://localhost:8000';

// Computed property for the currently displayed answer
const currentAnswer = computed(() => {
  if (answers.value.length > 0) {
    return answers.value[currentAnswerIndex.value];
  }
  return null;
});

// --- NEW COMPUTED PROPERTY for review constraint ---
const isReviewDisabledForCurrentK = computed(() => {
  const currentK = currentAnswerIndex.value + 1;
  // Constraint: can't post review for k > 1 unless k=1 has been reviewed
  return currentK > 1 && !reviewedKPositions.value.has(1);
});

// --- NEW: Computed property for the dot's class ---
const matcherDotClass = computed(() => {
  return useConcatMatcher.value ? 'dot-blue' : 'dot-yellow';
});

// --- Carousel Navigation Functions ---
const nextAnswer = () => {
  if (currentAnswerIndex.value < answers.value.length - 1) {
    currentAnswerIndex.value++;
  }
};

const prevAnswer = () => {
  if (currentAnswerIndex.value > 0) {
    currentAnswerIndex.value--;
  }
};

// --- Query Submission Function ---
const sendQuery = async () => {
  if (!userQuery.value.trim()) {
    queryError.value = "Please enter a query.";
    return;
  }

  loading.value = true;
  queryError.value = null;
  answers.value = [];
  currentAnswerIndex.value = 0;
  sessionId.value = null;
  queryId.value = null;
  reviewedKPositions.value.clear(); // --- NEW: Reset reviewed positions for new query ---

  try {
    const response = await fetch(`${API_BASE_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: userQuery.value,
        top_k: topK.value,
        metric: 'cosine',
        session_id: sessionId.value,
        use_concat_matcher: false, // Pass the selected option
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to fetch answers.');
    }

    const data = await response.json();
    answers.value = data.results;
    sessionId.value = data.session_id;
    queryId.value = data.query_id;

    if (answers.value.length === 0) {
        queryError.value = "No relevant answers found for your query.";
    }

  } catch (err) {
    queryError.value = err.message;
    console.error('Query error:', err);
  } finally {
    loading.value = false;
  }
};

// --- Event handler for review submission from AnswerCard ---
const handleReviewSubmitted = (reviewData) => {
  console.log('Review submitted from component:', reviewData);
  // --- NEW: Add the reviewed position to the set ---
  if (reviewData.position_in_results) {
    reviewedKPositions.value.add(reviewData.position_in_results);
  }
};

</script>

<style>
/* --- (CSS from previous app.vue, no changes needed here) --- */
/* Global Styles - these apply to the whole page */
body {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  margin: 0;
  padding: 0;
  background-color: #f0f2f5;
  color: #333;
  line-height: 1.6;
}

h1, h2, h3, h4, h5, h6 {
  color: #2c3e50;
  margin-bottom: 0.8em;
}

/* Container */
.container {
  max-width: 900px;
  margin: 40px auto;
  padding: 30px;
  background-color: #ffffff;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
  border-radius: 12px;
}

.text-center {
  text-align: center;
}

/* Query Section */
.query-section {
  display: flex;
  gap: 15px;
  margin-bottom: 40px;
  align-items: center;
}

.query-section input[type="text"] {
  flex-grow: 1;
  padding: 14px 20px;
  border: 1px solid #dcdcdc;
  border-radius: 8px;
  font-size: 1.1rem;
  transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

.query-section input[type="text"]:focus {
  border-color: #007bff;
  box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.25);
  outline: none;
}

.query-section button {
  padding: 14px 25px;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 1.1rem;
  font-weight: bold;
  transition: background-color 0.3s ease, transform 0.2s ease;
}

.query-section button:hover:not(:disabled) {
  background-color: #0056b3;
  transform: translateY(-2px);
}

.query-section button:disabled {
  background-color: #cccccc;
  cursor: not-allowed;
  opacity: 0.7;
}

/* Info and Error Messages */
.info-message {
  padding: 15px;
  margin-bottom: 25px;
  border-radius: 8px;
  font-size: 1rem;
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
  padding: 15px;
  margin-bottom: 25px;
  border-radius: 8px;
  font-size: 1rem;
  text-align: center;
}

.no-results-message {
    padding: 20px;
    background-color: #f8f9fa;
    border: 1px dashed #e9ecef;
    border-radius: 8px;
    color: #6c757d;
    font-size: 1.1rem;
    font-style: italic;
}

/* Answers Carousel */
.answers-carousel {
  margin-top: 30px;
  border: 1px solid #e0e0e0;
  border-radius: 12px;
  padding: 25px;
  background-color: #fdfdfd;
  box-shadow: inset 0 1px 5px rgba(0, 0, 0, 0.03);
}

.carousel-navigation {
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: 25px;
  gap: 20px;
}

.carousel-navigation button {
  background-color: #6c757d;
  color: white;
  border: none;
  border-radius: 8px;
  padding: 10px 20px;
  font-size: 1rem;
  cursor: pointer;
  transition: background-color 0.3s ease, transform 0.2s ease;
}

.carousel-navigation button:disabled {
  background-color: #cccccc;
  cursor: not-allowed;
  opacity: 0.6;
}

.carousel-navigation button:hover:not(:disabled) {
  background-color: #5a6268;
  transform: translateY(-2px);
}

.carousel-navigation .carousel-counter {
  font-size: 1.2rem;
  font-weight: bold;
  color: hsl(0, 46%, 62%);
}

/* Concat Indicator */
.concat-indicator {
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 25px;
  padding: 10px 15px;
  background-color: #f8f9fa; /* Light neutral background */
  border: 1px solid #e9ecef;
  border-radius: 8px;
  color: #6c757d; /* Darker grey text */
  font-size: 0.95rem;
  font-weight: 600;
  gap: 8px; /* Space between dot and text */
}

.status-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  display: inline-block;
  border: 1px solid rgba(0,0,0,0.1); /* Subtle border for visibility */
}

.dot-blue {
  background-color: #007bff !important; /* Blue for concatenated */
}

.dot-yellow {
  background-color: #ffc107 !important; /* Yellow for separate */
}
</style>