import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime, UTC # Changed datetime.utcnow to datetime.now(UTC) for timezone awareness
import uuid

# -----------------------------------------------------------------------------
# Database setup (stand-alone)
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
    timestamp = Column(DateTime, default=datetime.now(UTC)) # Use timezone-aware datetime
    returned_answer_ids = Column(Text, nullable=True)
    concat_option_active = Column(Boolean, nullable=False, default=True) # NEW: concat option

    reviews = relationship("UserReview", back_populates="query")


class UserReview(Base):
    __tablename__ = "user_reviews"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, default=lambda: str(uuid.uuid4()))
    answer_id = Column(Integer, nullable=False)
    review_code = Column(Integer, nullable=False)  # 1 = best, 5 = worst
    review_text = Column(Text, nullable=True)
    position_in_results = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=datetime.now(UTC)) # Use timezone-aware datetime
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
        # Fetch all reviews along with their associated query's concat_option_active status
        all_reviews_with_concat_info = (
            session.query(UserReview, UserQuery.concat_option_active)
            .join(UserQuery)
            .all()
        )

        # Separate reviews based on concat_option_active
        reviews_concat_true = [r for r, concat_active in all_reviews_with_concat_info if concat_active]
        reviews_concat_false = [r for r, concat_active in all_reviews_with_concat_info if not concat_active]

        report_configs = [
            {"label": "Concatenated Q&A (concat_q_and_a = True)", "reviews": reviews_concat_true, "filename": "report_concat_true.html"},
            {"label": "Separate Q&A (concat_q_and_a = False)", "reviews": reviews_concat_false, "filename": "report_concat_false.html"},
        ]

        for config in report_configs:
            label = config["label"]
            reviews_to_process = config["reviews"]
            out_file_name = config["filename"]

            total_reviews = len(reviews_to_process)

            if total_reviews == 0:
                print(f"No reviews found for '{label}'. Skipping report generation for this category.")
                continue

            # --------------------------------------------------------------
            # 1. Overall statistics
            # --------------------------------------------------------------
            sum_scores = sum(r.review_code for r in reviews_to_process)
            avg_overall_score = sum_scores / total_reviews

            score_counts = {sc: 0 for sc in range(1, 6)}
            for r in reviews_to_process:
                score_counts[r.review_code] += 1
            score_ratios = {s: c / total_reviews for s, c in score_counts.items()}

            # --------------------------------------------------------------
            # 2. Aggregate answers
            # --------------------------------------------------------------
            answers_data = {}
            for r in reviews_to_process:
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
<title>API Data Review Report - {label}</title>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    :root {{
        --brand-primary: #0d6efd;
        --brand-success: #198754;
        --brand-secondary: #6c757d;
        --text-dark: #212529;
        --text-light: #495057;
        --surface-bg: #f8f9fa;
        --content-bg: #ffffff;
        --border-color: #dee2e6;
        --shadow: 0 4px 8px rgba(0, 0, 0, 0.06);
        --radius: 8px;
    }}

    body {{
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: clamp(15px, 4vw, 40px);
        background: var(--surface-bg);
        color: var(--text-dark);
        line-height: 1.6;
    }}

    h1 {{ font-size: 2.2rem; font-weight: 700; color: var(--text-dark); text-align: center; margin-bottom: 30px; }}
    h2 {{ font-size: 1.7rem; font-weight: 600; color: var(--text-dark); border-bottom: 2px solid var(--border-color); padding-bottom: 12px; margin-bottom: 25px; }}
    h3 {{ font-size: 1.2rem; font-weight: 600; margin-top: 0; }}
    h4 {{ font-size: 1.1rem; font-weight: 600; color: var(--text-light); }}

    section {{
        background: var(--content-bg);
        padding: clamp(15px, 3vw, 25px);
        margin-bottom: 25px;
        border-radius: var(--radius);
        box-shadow: var(--shadow);
    }}

    /* --- Stats Section --- */
    .stats-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 20px;
    }}
    .stat-card {{
        background: var(--content-bg);
        border: 1px solid var(--border-color);
        padding: 20px;
        border-radius: var(--radius);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .stat-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.08);
    }}
    .stat-card h3 {{ color: var(--text-light); margin-bottom: 8px; font-size: 1rem; text-transform: uppercase; letter-spacing: .5px; }}
    .stat-card p {{ font-size: 2.25rem; font-weight: 700; color: var(--brand-primary); margin: 0; }}
    .stat-card:first-of-type p {{color: var(--brand-success);}}

    /* --- Filters Section --- */
    .filters {{
        background: var(--content-bg);
        padding: 20px;
        border-radius: var(--radius);
        margin-bottom: 25px;
        display: flex;
        flex-wrap: wrap;
        gap: 15px 25px;
        align-items: center;
    }}
    .filters label {{ font-weight: 600; margin-right: 5px; }}
    .filters select, .filters input[type='checkbox'] {{ margin-right: 5px; }}
    .filters select {{ padding: 8px; border-radius: 5px; border: 1px solid var(--border-color); background: #fff; }}
    .filters button {{
        padding: 10px 20px;
        color: #fff;
        background: var(--brand-primary);
        border: 0;
        border-radius: 5px;
        cursor: pointer;
        font-weight: 600;
        transition: all 0.2s ease;
    }}
    .filters button:hover {{ background: #0b5ed7; transform: translateY(-1px); }}
    .filters button:last-of-type {{ background-color: var(--brand-secondary); }}
    .filters button:last-of-type:hover {{ background-color: #5a6268; }}

    /* --- Answer Section --- */
    #answer-list {{
        display: grid;
        grid-template-columns: 1fr;
        gap: 20px;
    }}
    .answer-card {{
        background: var(--content-bg);
        border: 1px solid var(--border-color);
        margin-bottom: 0;
        padding: 20px;
        border-radius: var(--radius);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .answer-card:hover {{
        box-shadow: 0 8px 16px rgba(0,0,0,0.08);
    }}
    .answer-card h3 {{
        color: var(--brand-success);
        margin: 0 0 15px 0;
        padding-bottom: 15px;
        border-bottom: 1px solid var(--border-color);
    }}

    .review-card {{
        background: var(--surface-bg);
        border: 1px solid #e9ecef;
        margin-top: 15px;
        padding: 15px;
        border-radius: var(--radius);
        font-size: 0.95em;
    }}
    .review-card .code {{
        font-weight: 700;
        color: #b02a37;
    }}
    .review-card .text {{
        font-style: italic;
        color: var(--text-light);
        padding-left: 15px;
        border-left: 3px solid var(--border-color);
        margin-top: 10px;
        display: block; /* Ensures block-level treatment */
    }}
    .review-card p {{ margin: 5px 0; }}

    .hidden {{ display: none; }}
    .no-data {{ text-align: center; color: var(--text-light); padding: 50px; font-size: 1.2rem; }}
</style>
</head>
<body>
<h1>API Data Review Report - {label}</h1>
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

            html += """</div> </section>
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
            with open(out_file_name, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"Report generated successfully: {os.path.abspath(out_file_name)}")

    finally:
        session.close()


if __name__ == "__main__":
    generate_report()