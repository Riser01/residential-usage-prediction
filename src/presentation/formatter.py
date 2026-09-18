"""Formats prediction review output tables strictly conforming to Anacity PDF §3.4 & §3.5."""

import os
from datetime import datetime
from typing import Dict, List

import pandas as pd

from src.features.pipeline import DAY_NAMES


class PredictionFormatter:
    """Produces publication-grade comparison tables in CSV, Excel, and standalone HTML."""

    @staticmethod
    def build_comparison_table(
        meta_df: pd.DataFrame,
        y_pred_df: pd.DataFrame,
        y_true_df: pd.DataFrame,
        eval_details: pd.DataFrame,
    ) -> pd.DataFrame:
        """Constructs the canonical 4-column comparison table from PDF §3.5."""
        records = []
        n_samples = len(meta_df)

        for i in range(n_samples):
            rid = meta_df["resident_id"].iloc[i]
            past_summary = meta_df["past_bookings_summary"].iloc[i]

            # Prediction column string
            p_fac = y_pred_df["pred_facility"].iloc[i]
            p_day = y_pred_df["pred_usage_day"].iloc[i]
            p_time = y_pred_df["pred_usage_time"].iloc[i]
            p_nudge = y_pred_df["pred_nudge_str"].iloc[i]
            pred_str = f"{p_fac} / {p_day} / {p_time}\n{p_nudge}"

            # Actual column string
            a_fac = y_true_df["target_facility"].iloc[i]
            a_day = y_true_df["target_usage_day"].iloc[i]
            a_u_dt = pd.to_datetime(y_true_df["target_usage_timestamp"].iloc[i])
            a_b_dt = pd.to_datetime(y_true_df["target_booking_timestamp"].iloc[i])
            a_time = a_u_dt.strftime("%H:%M")
            a_book_day = DAY_NAMES[a_b_dt.weekday()]
            a_book_time = a_b_dt.strftime("%H:%M")
            actual_str = f"{a_fac} / {a_day} / {a_time}\nBooked {a_book_day} / {a_book_time}"

            # Match indicator
            match_str = eval_details["match_label"].iloc[i]
            total_m = eval_details["total_matched"].iloc[i]

            records.append(
                {
                    "Resident Reference": rid,
                    "PAST BOOKINGS": past_summary,
                    "PREDICTION": pred_str,
                    "ACTUAL": actual_str,
                    "MATCH": match_str.replace("\n", " "),
                    "Score": f"{total_m} of 4",
                    "Facility Match": "YES" if eval_details["fac_match"].iloc[i] else "NO",
                    "Day Match": "YES" if eval_details["day_match"].iloc[i] else "NO",
                    "Time Match": "YES" if eval_details["hour_match"].iloc[i] else "NO",
                    "Nudge Match": "YES" if eval_details["nudge_match"].iloc[i] else "NO",
                }
            )

        return pd.DataFrame(records)

    @staticmethod
    def export_all_formats(
        table_df: pd.DataFrame,
        metrics_summary: Dict[str, any],
        output_dir: str = "output",
    ) -> None:
        """Exports CSV, Excel, and standalone interactive HTML report."""
        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(output_dir, "prediction_review_table.csv")
        xlsx_path = os.path.join(output_dir, "prediction_review_table.xlsx")
        html_path = os.path.join(output_dir, "prediction_review.html")

        # 1. Export CSV
        table_df.to_csv(csv_path, index=False)

        # 2. Export Excel
        try:
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                table_df.to_excel(writer, sheet_name="Prediction Review", index=False)
                pd.DataFrame([metrics_summary]).to_excel(writer, sheet_name="Metrics Summary", index=False)
        except Exception as e:
            print(f"Warning: Excel export error: {e}")

        # 3. Export Standalone Zero-Dependency HTML Viewer
        PredictionFormatter._generate_html_viewer(table_df, metrics_summary, html_path)
        print(f"Exported prediction review deliverables to {output_dir}/")

    @staticmethod
    def _generate_html_viewer(
        table_df: pd.DataFrame, metrics_summary: Dict[str, any], output_file: str
    ) -> None:
        """Generates self-contained, responsive HTML table matching PDF §3.5 format."""
        # Top 100 rows for instant, lightweight browser rendering
        display_df = table_df.head(250)

        rows_html = []
        for _, r in display_df.iterrows():
            is_4_of_4 = "4 of 4" in r["Score"] and "YES" in r["MATCH"]
            badge_class = "badge-yes" if is_4_of_4 else "badge-no"
            past_html = r["PAST BOOKINGS"].replace("\n", "<br>")
            pred_html = r["PREDICTION"].replace("\n", "<br><strong>").replace("Nudge", "</strong>Nudge")
            act_html = r["ACTUAL"].replace("\n", "<br><strong>").replace("Booked", "</strong>Booked")

            rows_html.append(
                f"""
                <tr>
                    <td class="res-col"><strong>{r['Resident Reference']}</strong></td>
                    <td class="past-col">{past_html}</td>
                    <td class="pred-col">{pred_html}</td>
                    <td class="act-col">{act_html}</td>
                    <td class="match-col"><span class="badge {badge_class}">{r['MATCH']}</span></td>
                </tr>
                """
            )

        tbody_content = "\n".join(rows_html)

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Residential Usage Prediction System — Review Output</title>
    <style>
        :root {{
            --primary: #1e3a8a;
            --accent: #2563eb;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --border: #e2e8f0;
            --success-bg: #dcfce7;
            --success-text: #166534;
            --fail-bg: #fee2e2;
            --fail-text: #991b1b;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 24px;
            line-height: 1.5;
        }}
        .header {{
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            color: white;
            padding: 28px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-bottom: 24px;
        }}
        .header h1 {{ margin: 0 0 8px 0; font-size: 26px; }}
        .header p {{ margin: 0; opacity: 0.9; font-size: 14px; }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .metric-card {{
            background: var(--card-bg);
            padding: 16px;
            border-radius: 8px;
            border: 1px solid var(--border);
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .metric-val {{
            font-size: 24px;
            font-weight: 700;
            color: var(--accent);
            margin-top: 4px;
        }}
        .metric-lbl {{
            font-size: 12px;
            text-transform: uppercase;
            color: #64748b;
            letter-spacing: 0.5px;
        }}

        .search-bar {{
            background: white;
            padding: 14px;
            border-radius: 8px;
            border: 1px solid var(--border);
            margin-bottom: 20px;
            display: flex;
            gap: 12px;
            align-items: center;
        }}
        .search-input {{
            flex: 1;
            padding: 10px 14px;
            border: 1px solid var(--border);
            border-radius: 6px;
            font-size: 14px;
        }}
        .filter-select {{
            padding: 10px 14px;
            border: 1px solid var(--border);
            border-radius: 6px;
            font-size: 14px;
            background: white;
        }}

        .table-container {{
            background: var(--card-bg);
            border-radius: 12px;
            border: 1px solid var(--border);
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: left;
        }}
        th {{
            background: #f1f5f9;
            color: #475569;
            padding: 12px 16px;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
            border-bottom: 2px solid var(--border);
        }}
        td {{
            padding: 14px 16px;
            border-bottom: 1px solid var(--border);
            vertical-align: top;
        }}
        tr:hover {{ background-color: #f8fafc; }}
        
        .res-col {{ width: 10%; font-weight: 600; color: #0f172a; }}
        .past-col {{ width: 32%; color: #475569; font-family: monospace; font-size: 12px; }}
        .pred-col {{ width: 25%; font-family: monospace; font-size: 12px; color: #1e40af; }}
        .act-col {{ width: 23%; font-family: monospace; font-size: 12px; color: #0f172a; }}
        .match-col {{ width: 10%; text-align: center; }}

        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 11px;
            text-align: center;
        }}
        .badge-yes {{ background: var(--success-bg); color: var(--success-text); }}
        .badge-no {{ background: var(--fail-bg); color: var(--fail-text); }}
    </style>
</head>
<body>

    <div class="header">
        <h1>Residential Usage Prediction System</h1>
        <p>Prediction Review Table conforming to Assignment §3.4 & §3.5 | Chronological Unseen Holdout Evaluation</p>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-lbl">Facility Accuracy</div>
            <div class="metric-val">{metrics_summary.get('facility_accuracy', 0.0) * 100:.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-lbl">Usage Day Accuracy</div>
            <div class="metric-val">{metrics_summary.get('usage_day_accuracy', 0.0) * 100:.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-lbl">Hour ±1hr Accuracy</div>
            <div class="metric-val">{metrics_summary.get('usage_hour_within_1hr_accuracy', 0.0) * 100:.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-lbl">Proactive Nudge Rate</div>
            <div class="metric-val">{metrics_summary.get('nudge_proactive_actionability_rate', 0.0) * 100:.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-lbl">Exact 4 of 4 Matches</div>
            <div class="metric-val">{metrics_summary.get('exact_4_of_4_match_rate', 0.0) * 100:.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-lbl">Avg Outputs Matched</div>
            <div class="metric-val">{metrics_summary.get('average_outputs_matched', 0.0):.2f} / 4.0</div>
        </div>
    </div>

    <div class="search-bar">
        <input type="text" id="searchInput" class="search-input" placeholder="Search by resident ID, facility, or day..." onkeyup="filterTable()">
        <select id="matchFilter" class="filter-select" onchange="filterTable()">
            <option value="ALL">All Match Statuses</option>
            <option value="YES">Exact Matches (4 of 4)</option>
            <option value="NO">Partial Matches (< 4)</option>
        </select>
    </div>

    <div class="table-container">
        <table id="reviewTable">
            <thead>
                <tr>
                    <th>Resident</th>
                    <th>PAST BOOKINGS<br><small style="text-transform:none;color:#64748b">facility / day / use time / booked at</small></th>
                    <th>PREDICTION<br><small style="text-transform:none;color:#64748b">facility / day / use time / nudge time</small></th>
                    <th>ACTUAL<br><small style="text-transform:none;color:#64748b">facility / day / use time / booked at</small></th>
                    <th>MATCH</th>
                </tr>
            </thead>
            <tbody>
                {tbody_content}
            </tbody>
        </table>
    </div>

    <script>
        function filterTable() {{
            var input = document.getElementById("searchInput").value.toUpperCase();
            var matchFilter = document.getElementById("matchFilter").value;
            var table = document.getElementById("reviewTable");
            var tr = table.getElementsByTagName("tr");

            for (var i = 1; i < tr.length; i++) {{
                var text = tr[i].textContent || tr[i].innerText;
                var matchCol = tr[i].getElementsByTagName("td")[4];
                var matchText = matchCol ? (matchCol.textContent || matchCol.innerText) : "";

                var textMatch = text.toUpperCase().indexOf(input) > -1;
                var statusMatch = true;
                if (matchFilter === "YES") {{
                    statusMatch = matchText.indexOf("YES") > -1;
                }} else if (matchFilter === "NO") {{
                    statusMatch = matchText.indexOf("NO") > -1;
                }}

                if (textMatch && statusMatch) {{
                    tr[i].style.display = "";
                }} else {{
                    tr[i].style.display = "none";
                }}
            }}
        }}
    </script>
</body>
</html>
"""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_template)
