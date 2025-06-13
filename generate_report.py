# report.py
import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import uuid

# -----------------------------------------------------------------------------
# Database setup (stand‑alone)
# -----------------------------------------------------------------------------
DATABASE_URL = "sqlite:///data/api_data.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class UserQuery(Base):
    __tablename__ = "user_queries"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, default=lambda: str(uuid.uuid4()))
    query_text = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    returned_answer_ids = Column(Text, nullable=True)

    reviews = relationship("UserReview", back_populates="query")


class UserReview(Base):
    __tablename__ = "user_reviews"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, default=lambda: str(uuid.uuid4()))
    answer_id = Column(Integer, nullable=False)
    review_code = Column(Integer, nullable=False)  # 1 = best, 5 = worst
    review_text = Column(Text, nullable=True)
    position_in_results = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    query_id = Column(Integer, ForeignKey("user_queries.id"), nullable=False)

    query = relationship("UserQuery", back_populates="reviews")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def create_db_and_tables() -> None:
    """Ensure tables exist before running a report."""
    Base.metadata.create_all(bind=engine)
    print("Database tables created or already exist.")


# -----------------------------------------------------------------------------
# Report generation
# -----------------------------------------------------------------------------

def generate_report() -> None:
    create_db_and_tables()
    session = SessionLocal()

    try:
        all_reviews = session.query(UserReview).all()
        total_reviews = len(all_reviews)

        if total_reviews == 0:
            print("No reviews found in the database. Exiting.")
            return

        # --------------------------------------------------------------
        # 1. Overall statistics
        # --------------------------------------------------------------
        sum_scores = sum(r.review_code for r in all_reviews)
        avg_overall_score = sum_scores / total_reviews

        score_counts = {sc: 0 for sc in range(1, 6)}
        for r in all_reviews:
            score_counts[r.review_code] += 1
        score_ratios = {s: c / total_reviews for s, c in score_counts.items()}

        # --------------------------------------------------------------
        # 2. Aggregate answers
        # --------------------------------------------------------------
        answers_data = {}
        for r in all_reviews:
            data = answers_data.setdefault(
                r.answer_id,
                {
                    "reviews": [],
                    "sum_scores": 0,
                    "count_scores": 0,
                    "codes": set(),
                },
            )
            data["reviews"].append(
                {
                    "id": r.id,
                    "review_code": r.review_code,
                    "review_text": r.review_text,
                    "position": r.position_in_results,
                    "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            data["sum_scores"] += r.review_code
            data["count_scores"] += 1
            data["codes"].add(r.review_code)

        answers_list = [
            {
                "answer_id": aid,
                "avg": d["sum_scores"] / d["count_scores"],
                "reviews": d["reviews"],
                "codes": d["codes"],
            }
            for aid, d in answers_data.items()
        ]
        answers_list.sort(key=lambda x: x["avg"])  # lower is better

        # --------------------------------------------------------------
        # 3. Build HTML
        # --------------------------------------------------------------
        html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"UTF-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
<title>API Data Review Report</title>
<style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background: #f4f4f4; color: #333; }}
    h1, h2 {{ color: #0056b3; }}
    section {{ background: #fff; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,.1); }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; }}
    .stat-card {{ background: #e9f5ff; border: 1px solid #cce0ff; padding: 15px; border-radius: 5px; text-align: center; }}
    .stat-card h3 {{ margin-top: 0; color: #004085; }}
    .filters {{ background: #f0f8ff; padding: 15px; border-radius: 8px; margin-bottom: 20px; display: flex; flex-wrap: wrap; gap: 15px; align-items: center; }}
    .filters label {{ font-weight: 700; margin-right: 5px; }}
    .filters button {{ padding: 8px 15px; background: #007bff; color: #fff; border: 0; border-radius: 5px; cursor: pointer; transition: background .2s; }}
    .filters button:hover {{ background: #0056b3; }}
    .answer-card {{ background: #fff; border: 1px solid #dcdcdc; margin-bottom: 15px; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,.08); transition: transform .2s; }}
    .answer-card:hover {{ transform: translateY(-3px); }}
    .answer-card h3 {{ color: #28a745; margin-top: 0; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 10px; }}
    .review-card {{ background: #f8f9fa; border: 1px solid #e2e6ea; margin: 8px 0; padding: 10px; border-radius: 5px; font-size: .9em; }}
    .review-card .code {{ font-weight: 700; color: #dc3545; }}
    .review-card .text {{ font-style: italic; color: #666; }}
    .hidden {{ display: none; }}
    .no-data {{ text-align: center; color: #888; padding: 50px; }}
</style>
</head>
<body>
<h1>API Data Review Report</h1>
<section id=\"stats\"><h2>Overall Statistics</h2><div class=\"stats-grid\">"""
        html += (
            f"<div class='stat-card'><h3>Total Reviews</h3><p>{total_reviews}</p></div>"
            f"<div class='stat-card'><h3>Average Overall Score (1=Best, 5=Worst)</h3><p>{avg_overall_score:.2f}</p></div>"
        )
        for score in range(1, 6):
            pct = score_ratios.get(score, 0) * 100
            html += f"<div class='stat-card'><h3>Score {score} Ratio</h3><p>{score_counts[score]} / {total_reviews} = {pct:.2f}%</p></div>"
        html += "</div></section>"

        # Filters + answers
        html += """
<section id=\"answer-section\">
  <h2>Answer Reviews</h2>
  <div class=\"filters\" id=\"filters\">
    <div>
      <label for=\"avg-score-filter\">Average Score:</label>
      <select id=\"avg-score-filter\">
        <option value='all'>All</option>
        <option value='1.0-1.9'>1.0 ‑ 1.9</option>
        <option value='2.0-2.9'>2.0 ‑ 2.9</option>
        <option value='3.0-3.9'>3.0 ‑ 3.9</option>
        <option value='4.0-5.0'>4.0 ‑ 5.0</option>
      </select>
    </div>
    <div>
      <label>Contains Review Score:</label>
      <input type='checkbox' id='filter-1' value='1'> <label for='filter-1'>1 (Best)</label>
      <input type='checkbox' id='filter-2' value='2'> <label for='filter-2'>2</label>
      <input type='checkbox' id='filter-3' value='3'> <label for='filter-3'>3</label>
      <input type='checkbox' id='filter-4' value='4'> <label for='filter-4'>4</label>
      <input type='checkbox' id='filter-5' value='5'> <label for='filter-5'>5 (Worst)</label>
    </div>
    <button onclick='applyFilters()'>Apply Filters</button>
    <button onclick='clearFilters()'>Clear Filters</button>
  </div>
  <div id='answer-list'>"""

        if not answers_list:
            html += "<p class='no-data'>No answers with reviews found.</p>"
        else:
            for ans in answers_list:
                attr_str = " ".join(
                    f"data-has-review-{code}='true'" for code in ans["codes"]
                )
                html += (
                    f"<div class='answer-card' data-avg-score='{ans['avg']:.2f}' {attr_str}>"
                    f"<h3>Answer ID: {ans['answer_id']} (Average Score: {ans['avg']:.2f})</h3>"
                    "<h4>Individual Reviews:</h4>"
                )
                for rev in ans["reviews"]:
                    html += (
                        "<div class='review-card'>"
                        f"<p>Review ID: {rev['id']}</p>"
                        f"<p>Score: <span class='code'>{rev['review_code']}</span></p>"
                        f"<p>Position in Results: {rev['position'] if rev['position'] is not None else 'N/A'}</p>"
                        f"<p class='text'>Text: {rev['review_text'] or 'No text provided.'}</p>"
                        f"<p>Timestamp: {rev['timestamp']}</p>"
                        "</div>"
                    )
                html += "</div>"  # /answer-card

        html += """</div> <!-- /answer-list -->
</section>
<script>
function applyFilters() {
  const avgVal = document.getElementById('avg-score-filter').value;
  const selectedCodes = Array.from(
      document.querySelectorAll('#filters input[type="checkbox"]:checked'))
      .map(cb => cb.value);

  document.querySelectorAll('.answer-card').forEach(card => {
    let show = true;
    const score = parseFloat(card.dataset.avgScore);

    // Average score filter
    if (avgVal !== 'all') {
      const [min, max] = avgVal.split('-').map(parseFloat);
      if (score < (min - 0.001) || score > (max + 0.001)) show = false;
    }

    // Review code filter (uses data attribute presence)
    if (selectedCodes.length && show) {
      show = selectedCodes.some(code => card.hasAttribute(`data-has-review-${code}`));
    }

    card.classList.toggle('hidden', !show);
  });
}

function clearFilters() {
  document.getElementById('avg-score-filter').value = 'all';
  document.querySelectorAll('#filters input[type="checkbox"]').forEach(cb => cb.checked = false);
  applyFilters();
}
</script>
</body>
</html>"""

        # --------------------------------------------------------------
        # 4. Write
        # --------------------------------------------------------------
        out_file = "report.html"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Report generated successfully: {os.path.abspath(out_file)}")

    finally:
        session.close()


if __name__ == "__main__":
    generate_report()
