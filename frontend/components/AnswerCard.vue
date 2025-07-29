<template>
  <div class="answer-card">
    <div v-if="debug">
      <h3>Matched Prompt:</h3>
      <p class="prompt-text">{{ answer.matched_prompt }}</p>

      <div class="answer-meta">
        <p>Score: <strong>{{ answer.score.toFixed(4) }}</strong> ({{ answer.metric }})</p>
        <p>Question ID: <strong>{{ answer.question_id }}</strong>, Answer ID: <strong>{{ answer.answer_id }}</strong></p>
        <p>This is answer **{{ currentKPosition }}** in the list.</p>
      </div>
    </div>

    <h4>Response:</h4>
    <p class="response-text">{{ answer.response }} {{ answer.instruction }}</p>

    <div class="review-section">
      <h4>Rate this answer:</h4>
      <div v-if="reviewDisabled" class="info-message error-message constraint-message">
        Please review the first answer (Answer #1) before reviewing this one.
      </div>
      <div class="review-buttons">
        <button
          v-for="n in 4"
          :key="n"
          @click="selectReview(n)"
          :disabled="reviewDisabled || submittingReview" :class="{
            'selected': selectedReviewCode === n,
            'btn-good': n === 1,
            'btn-okay': n === 2,
            'btn-bad': n === 3 || n === 4,
            'text-white': n !== 2
          }"
        >
          {{ getReviewButtonText(n) }}
        </button>
      </div>
      <textarea
        v-model="reviewText"
        placeholder="Optional review text (e.g., 'This was helpful but slightly off')..."
        :disabled="reviewDisabled || submittingReview" ></textarea>
      <button @click="submitReview" :disabled="reviewDisabled || !selectedReviewCode || submittingReview" class="btn-submit-review">
        <span v-if="!submittingReview">Send Review</span>
        <span v-else>Submitting...</span>
      </button>
      <div v-if="reviewMessage" :class="['info-message', { 'error-message': reviewError }]">{{ reviewMessage }}</div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue';

const props = defineProps({
  answer: {
    type: Object,
    required: true
  },
  sessionId: {
    type: String,
    required: true
  },
  currentKPosition: { // 1-indexed position in the returned list
    type: Number,
    required: true
  },
  reviewDisabled: { // Flag from parent to disable review controls
    type: Boolean,
    default: false
  },
  queryId: {
    type: Number,
    required: true // Query ID is now mandatory for reviews
  },
  // --- NEW PROP: Debug Mode ---
  debug: {
    type: Boolean,
    default: false // Default to false (production mode)
  }
});

const emit = defineEmits(['reviewSubmitted']);

// Reactive state for review form
const selectedReviewCode = ref(null);
const reviewText = ref('');
const submittingReview = ref(false);
const reviewMessage = ref(null);
const reviewError = ref(false);

// FastAPI Backend URL
const API_BASE_URL = 'http://localhost:8000';

// Watch for changes in the 'answer' prop to reset the form
watch(() => props.answer, () => {
  resetReviewForm();
}, { deep: true });

// Functions for review submission
const getReviewButtonText = (code) => {
  switch (code) {
    case 1: return 'Perfect';
    case 2: return 'Useful';
    case 3: return 'Irrelevant';
    case 4: return 'Incorrect';
    default: return '';
  }
};

const selectReview = (code) => {
  selectedReviewCode.value = code;
  reviewMessage.value = null; // Clear any previous review message
  reviewError.value = false;
};

const resetReviewForm = () => {
  selectedReviewCode.value = null;
  reviewText.value = '';
  reviewMessage.value = null;
  reviewError.value = false;
};

const submitReview = async () => {
  if (props.reviewDisabled) {
    reviewMessage.value = 'Cannot submit review: Please review the first answer (Answer #1) first.';
    reviewError.value = true;
    return;
  }
  if (!selectedReviewCode.value) {
    reviewMessage.value = 'Please select a rating.';
    reviewError.value = true;
    return;
  }
  if (!props.sessionId) {
    reviewMessage.value = 'Error: Session ID missing. Cannot submit review.';
    reviewError.value = true;
    return;
  }
  if (!props.answer) {
    reviewMessage.value = 'Error: No answer selected for review.';
    reviewError.value = true;
    return;
  }
  if (!props.queryId) {
    reviewMessage.value = 'Error: Query ID missing. Cannot submit review.';
    reviewError.value = true;
    return;
  }

  submittingReview.value = true;
  reviewMessage.value = null; // Clear previous messages
  reviewError.value = false;

  try {
    const payload = {
      session_id: props.sessionId,
      answer_id: props.answer.answer_id,
      review_code: selectedReviewCode.value,
      review_text: reviewText.value || null,
      position_in_results: props.currentKPosition,
      query_id: props.queryId
    };

    const response = await fetch(`${API_BASE_URL}/review`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to submit review.');
    }

    const data = await response.json();
    reviewMessage.value = data.message + ` (Review ID: ${data.review_id})`;
    emit('reviewSubmitted', { ...data, position_in_results: props.currentKPosition });
    resetReviewForm();

  } catch (err) {
    reviewMessage.value = `Error submitting review: ${err.message}`;
    reviewError.value = true;
    console.error('Review submission error:', err);
  } finally {
    submittingReview.value = false;
  }
};
</script>

<style scoped>
/* --- (CSS from previous AnswerCard.vue, with minor tweaks for disabled states) --- */
.answer-card {
  border: 1px solid #e9ecef;
  padding: 30px;
  border-radius: 10px;
  background-color: #ffffff;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

.answer-card h3 {
  color: #007bff;
  font-size: 1.4rem;
  margin-bottom: 10px;
}
.answer-card h4 {
    color: #343a40;
    font-size: 1.2rem;
    margin-top: 20px;
    margin-bottom: 8px;
}

.answer-card .prompt-text,
.answer-card .response-text {
  background-color: #f8f9fa;
  padding: 15px;
  border-radius: 6px;
  border: 1px solid #e9ecef;
  margin-bottom: 20px;
  word-wrap: break-word; /* Ensure long words break */
  white-space: pre-wrap; /* Preserve formatting for newlines if any */
}

.answer-card .answer-meta p {
  font-size: 0.95rem;
  color: #6c757d;
  margin-bottom: 5px;
}
.answer-card .answer-meta strong {
    color: #495057;
}

/* Review Section */
.review-section {
  margin-top: 35px;
  padding-top: 25px;
  border-top: 1px dashed #ced4da;
}

.review-section h4 {
  margin-bottom: 20px;
  color: #555;
  font-size: 1.2rem;
}

.review-buttons {
  display: flex;
  gap: 12px;
  margin-bottom: 20px;
  flex-wrap: wrap; /* Allow buttons to wrap */
}

.review-buttons button {
  padding: 10px 18px;
  border: 2px solid transparent; /* Default transparent border */
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.3s ease;
  font-weight: bold;
  font-size: 0.95rem;
  flex: 1 1 auto; /* Allow buttons to grow/shrink */
  min-width: 90px; /* Minimum width to prevent squishing */
  text-align: center;
}

/* Specific colors for review buttons */
.review-buttons .btn-good { background-color: #28a745; color: white; } /* Green */
.review-buttons .btn-okay { background-color: #ffc107; color: #343a40; } /* Yellow-Orange, dark text */
.review-buttons .btn-bad { background-color: #dc3545; color: white; } /* Red */

.review-buttons button:hover:not(.selected):not(:disabled) { /* Adjusted hover for disabled */
  filter: brightness(1.1); /* Slightly brighter on hover */
  transform: translateY(-1px);
}

.review-buttons button.selected {
  border-color: #007bff; /* Highlight selected button with blue border */
  box-shadow: 0 0 0 4px rgba(0, 123, 255, 0.2); /* Soft blue glow */
  transform: translateY(-2px);
}

.review-buttons button:disabled { /* Styling for disabled buttons */
  opacity: 0.6;
  cursor: not-allowed;
  filter: grayscale(50%);
}


textarea {
  width: 100%;
  padding: 12px 15px;
  border: 1px solid #ccc;
  border-radius: 8px;
  min-height: 100px;
  max-width: 100%;
  box-sizing: border-box; /* Include padding in width */
  margin-bottom: 20px;
  font-size: 1rem;
  resize: vertical; /* Allow vertical resizing */
}
textarea:disabled { /* Styling for disabled textarea */
  background-color: #f0f0f0;
  cursor: not-allowed;
}

.btn-submit-review {
  width: 100%;
  padding: 15px 20px;
  background-color: #17a2b8; /* A nice blue for submission */
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 1.15rem;
  font-weight: bold;
  transition: background-color 0.3s ease, transform 0.2s ease;
}

.btn-submit-review:hover:not(:disabled) {
  background-color: #138496;
  transform: translateY(-2px);
}

.btn-submit-review:disabled {
  background-color: #cccccc;
  cursor: not-allowed;
  opacity: 0.7;
}

/* Info and Error Messages specific to component */
.info-message {
  padding: 10px;
  margin-top: 15px;
  border-radius: 6px;
  font-size: 0.9rem;
  text-align: center;
  background-color: #d4edda; /* Light green */
  color: #155724; /* Dark green */
  border: 1px solid #c3e6cb;
}

.error-message {
  background-color: #f8d7da; /* Light red */
  color: #721c24; /* Dark red */
  border: 1px solid #f5c6cb;
}

.constraint-message {
  background-color: #fff3cd; /* Light yellow for warning */
  color: #856404; /* Dark yellow for text */
  border-color: #ffeeba;
  margin-bottom: 20px;
}
</style>