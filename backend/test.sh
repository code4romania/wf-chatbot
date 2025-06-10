#!/bin/bash

# --- Configuration ---
API_BASE_URL="http://127.0.0.1:8000" # Your API's base URL
HEADER="Content-Type: application/json"

echo "--- Starting Manual API Test Script ---"
echo ""

# --- Test 1: Root Endpoint ---
echo "1. Testing GET /"
curl -X GET "${API_BASE_URL}/"
echo ""
echo "---------------------------------------"
echo ""

# --- Test 2: Query for a prompt (top_k=1, generated session_id) ---
echo "2. Testing POST /query (single match, generated session_id)"
QUERY_PAYLOAD_1='{
  "query": "What are the common symptoms of influenza?",
  "top_k": 1,
  "metric": "cosine"
}'
echo "Sending query: ${QUERY_PAYLOAD_1}"
RESPONSE_1=$(curl -s -X POST "${API_BASE_URL}/query" \
  -H "${HEADER}" \
  -d "${QUERY_PAYLOAD_1}")

echo "Response: ${RESPONSE_1}"
SESSION_ID_1=$(echo "${RESPONSE_1}" | jq -r '.session_id') # Use jq to parse JSON
ANSWER_ID_1=$(echo "${RESPONSE_1}" | jq -r '.results[0].answer_id')
echo "Extracted Session ID: ${SESSION_ID_1}"
echo "Extracted Answer ID: ${ANSWER_ID_1}"
echo ""
echo "---------------------------------------"
echo ""

# --- Test 3: Submit a review for the first query's answer (Good review) ---
echo "3. Testing POST /review (Good review for first query)"
REVIEW_PAYLOAD_1='{
  "session_id": "'"${SESSION_ID_1}"'",
  "answer_id": '${ANSWER_ID_1}',
  "review_code": 1,
  "review_text": "This answer was perfect and very helpful!"
}'
echo "Sending review: ${REVIEW_PAYLOAD_1}"
curl -X POST "${API_BASE_URL}/review" \
  -H "${HEADER}" \
  -d "${REVIEW_PAYLOAD_1}"
echo ""
echo "---------------------------------------"
echo ""

# --- Test 4: Query for a prompt (top_k=3, specific session_id) ---
echo "4. Testing POST /query (multiple matches, provided session_id)"
CUSTOM_SESSION_ID="my_custom_browser_session_123" # A unique session ID you define
QUERY_PAYLOAD_2='{
  "query": "How can I apply for a new travel document?",
  "top_k": 3,
  "metric": "cosine",
  "session_id": "'"${CUSTOM_SESSION_ID}"'"
}'
echo "Sending query: ${QUERY_PAYLOAD_2}"
RESPONSE_2=$(curl -s -X POST "${API_BASE_URL}/query" \
  -H "${HEADER}" \
  -d "${QUERY_PAYLOAD_2}")

echo "Response: ${RESPONSE_2}"
ANSWER_IDS_2=$(echo "${RESPONSE_2}" | jq -r '[.results[].answer_id] | join(",")')
echo "Extracted Answer IDs: ${ANSWER_IDS_2}"
echo ""
echo "---------------------------------------"
echo ""

# --- Test 5: Submit a review for one of the answers from the second query (Okay review) ---
echo "5. Testing POST /review (Okay review for a specific answer from second query)"
# Pick one of the answer IDs from ANSWER_IDS_2 to review, e.g., the first one
# For this example, let's assume the first answer ID from the mock data is 102
# If your mock data order varies, you might need to manually inspect RESPONSE_2
ANSWER_TO_REVIEW_2="102" # Assuming one of the matched answer IDs is 102 from "How to apply for a new passport?"
REVIEW_PAYLOAD_2='{
  "session_id": "'"${CUSTOM_SESSION_ID}"'",
  "answer_id": '${ANSWER_TO_REVIEW_2}',
  "review_code": 2,
  "review_text": "The answer was okay, but could be more detailed."
}'
echo "Sending review: ${REVIEW_PAYLOAD_2}"
curl -X POST "${API_BASE_URL}/review" \
  -H "${HEADER}" \
  -d "${REVIEW_PAYLOAD_2}"
echo ""
echo "---------------------------------------"
echo ""

# --- Test 6: Submit another review for a *different* answer from the second query (Worst review) ---
echo "6. Testing POST /review (Worst review for another answer from second query)"
# Assuming another matched answer ID is 104 from "Where can I find information about visa applications?"
ANSWER_TO_REVIEW_3="104"
REVIEW_PAYLOAD_3='{
  "session_id": "'"${CUSTOM_SESSION_ID}"'",
  "answer_id": '${ANSWER_TO_REVIEW_3}',
  "review_code": 5,
  "review_text": "This was not relevant at all."
}'
echo "Sending review: ${REVIEW_PAYLOAD_3}"
curl -X POST "${API_BASE_URL}/review" \
  -H "${HEADER}" \
  -d "${REVIEW_PAYLOAD_3}"
echo ""
echo "---------------------------------------"
echo ""

echo "--- Manual API Test Script Finished ---"