"""
===============================================================================
Project:       VLEO Mission Analysis and Platform Analysis for an EP-driven,
               EO Satellite
Collaboration: In collaboration with ArianeGroup
Institution:   Cranfield University
Author:        Arnau Pascual
Year:          2026
===============================================================================
File:          generate_report.py
Description:
    Automated HTML report compilation and figure embedding for validation outputs.
===============================================================================
"""

import os
import csv
import glob
import re
import sys
from collections import defaultdict
from datetime import datetime
from itertools import groupby

# Ensure we can import from src
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from validations.orbital_decay_validation import SPACECRAFT_REGISTRY


def generate_report():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_base = os.path.join(project_root, "results")
    data_dir = os.path.join(results_base, "results_decay_validations")
    report_path = os.path.join(results_base, "validation_report.html")

    if not os.path.exists(data_dir):
        print(f"No data directory found at {data_dir}. Run the validation script first.")
        return

    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        print("No validation results found. Run the validation script first.")
        return

    html_content = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "    <meta charset='UTF-8'>",
        "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "    <title>Orbital Decay Validation Report</title>",
        "    <style>",
        "        :root {",
        "            --bg-color: #121212;",
        "            --text-color: #e0e0e0;",
        "            --card-bg: #1e1e1e;",
        "            --border-color: #333;",
        "            --accent-color: #00bcd4;",
        "        }",
        "        body {",
        "            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;",
        "            background-color: var(--bg-color);",
        "            color: var(--text-color);",
        "            margin: 0;",
        "            padding: 20px;",
        "            line-height: 1.6;",
        "        }",
        "        h1 {",
        "            text-align: center;",
        "            color: var(--accent-color);",
        "            margin-bottom: 40px;",
        "            font-weight: 300;",
        "            letter-spacing: 2px;",
        "        }",
        "        .validation-card {",
        "            background-color: var(--card-bg);",
        "            border: 1px solid var(--border-color);",
        "            border-radius: 8px;",
        "            padding: 25px;",
        "            margin-bottom: 40px;",
        "            box-shadow: 0 4px 6px rgba(0,0,0,0.3);",
        "        }",
        "        .validation-title {",
        "            font-size: 1.5em;",
        "            border-bottom: 2px solid var(--accent-color);",
        "            padding-bottom: 10px;",
        "            margin-bottom: 20px;",
        "            color: #fff;",
        "        }",
        "        table {",
        "            width: 100%;",
        "            border-collapse: collapse;",
        "            margin-bottom: 30px;",
        "        }",
        "        th, td {",
        "            padding: 12px 15px;",
        "            text-align: left;",
        "            border-bottom: 1px solid var(--border-color);",
        "        }",
        "        th {",
        "            background-color: #2c2c2c;",
        "            color: var(--accent-color);",
        "            font-weight: 600;",
        "        }",
        "        tr:hover {",
        "            background-color: #2a2a2a;",
        "        }",
        "        .image-gallery {",
        "            display: flex;",
        "            flex-wrap: wrap;",
        "            gap: 20px;",
        "            justify-content: center;",
        "        }",
        "        .image-gallery img {",
        "            max-width: 100%;",
        "            height: auto;",
        "            border-radius: 4px;",
        "            border: 1px solid var(--border-color);",
        "            box-shadow: 0 2px 4px rgba(0,0,0,0.5);",
        "        }",
        "        .image-card {",
        "            flex: 1 1 45%;",
        "            min-width: 300px;",
        "        }",
        "        .timestamp {",
        "            text-align: center;",
        "            color: #888;",
        "            font-size: 0.9em;",
        "            margin-top: 50px;",
        "        }",
        "    </style>",
        "</head>",
        "<body>",
        "    <h1>ORBITAL DECAY VALIDATION REPORT</h1>"
    ]


    # Auto-detect all missions present in the CSV files by matching against the
    # full SPACECRAFT_REGISTRY key list, so any new mission is picked up automatically.
    all_registry_keys = list(SPACECRAFT_REGISTRY.keys())
    unique_missions = set()
    for csv_file in csv_files:
        bn = os.path.basename(csv_file).replace("_validation.csv", "").lower()
        for key in all_registry_keys:
            if bn.startswith(key):
                unique_missions.add(key)
                break

    if unique_missions:
        html_content.append("    <div class='validation-card'>")
        html_content.append("        <div class='validation-title'>Spacecraft Physical Parameters</div>")
        html_content.append("        <table>")
        html_content.append("            <thead><tr>")
        html_content.append("                <th>Spacecraft</th>")
        html_content.append("                <th>Mass (kg)</th>")
        html_content.append("                <th>Cross-Section (m²)</th>")
        html_content.append("                <th>Drag Coeff Min</th>")
        html_content.append("                <th>Drag Coeff Max</th>")
        html_content.append("            </tr></thead>")
        html_content.append("            <tbody>")
        for m_key in sorted(unique_missions):
            params = SPACECRAFT_REGISTRY.get(m_key)
            if params:
                html_content.append("            <tr>")
                html_content.append(f"                <td>{m_key.upper()}</td>")
                html_content.append(f"                <td>{params.get('mass', 'N/A')}</td>")
                html_content.append(f"                <td>{params.get('cross_section', 'N/A')}</td>")
                html_content.append(f"                <td>{params.get('drag_coeff_min', 'N/A')}</td>")
                html_content.append(f"                <td>{params.get('drag_coeff_max', 'N/A')}</td>")
                html_content.append("            </tr>")
        html_content.append("            </tbody>")
        html_content.append("        </table>")
        html_content.append("    </div>")


    # ============================================================
    # MODEL PERFORMANCE SUMMARY (pivoted: rows=model×bound, cols=mission)
    # ============================================================
    MODEL_ORDER   = ["NRLMSISE-00", "JB2008", "DTM2000", "Harris-Priester"]
    all_data_rows = []

    for csv_file in csv_files:
        basename_csv = os.path.basename(csv_file).replace("_validation.csv", "").lower()
        file_mission = basename_csv.split("_")[0].upper()
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    label = row.get("Model (Bound)", "")
                    m = re.match(r"^(\S+)\s+(.+?)\s+\(Cd\s+([\d.]+)\)", label)
                    if not m:
                        continue
                    try:
                        all_data_rows.append({
                            "file_mission": file_mission,
                            "model":        m.group(2),
                            "cd_value":     float(m.group(3)),
                            "err_days":     float(row.get("Err (d)", 0)),
                            "err_pct":      float(row.get("Err (%)", 0)),
                        })
                    except ValueError:
                        pass
        except Exception as e:
            print(f"Warning: could not parse {csv_file}: {e}")

    # Label rows as Cd Min / Cd Max within each (file_mission, model) pair
    def _label_bound(rows):
        key_fn = lambda r: (r["file_mission"], r["model"])
        for _, grp in groupby(sorted(rows, key=key_fn), key=key_fn):
            grp = list(grp)
            lo  = min(r["cd_value"] for r in grp)
            for r in grp:
                r["bound"] = "Cd Min" if r["cd_value"] == lo else "Cd Max"
        return rows

    all_data_rows = _label_bound(all_data_rows)

    # Build pivoted lookup: mission_data[(model, bound)][mission] -> {err_days, err_pct}
    mission_data = defaultdict(dict)
    for r in all_data_rows:
        mission_data[(r["model"], r["bound"])][r["file_mission"]] = {
            "err_days": r["err_days"],
            "err_pct":  r["err_pct"],
        }

    # Sorted list of all unique missions present
    all_missions = sorted(set(r["file_mission"] for r in all_data_rows))

    def _color(val, lo=5.0, hi=20.0):
        a = abs(val)
        if   a <= lo: return "#4caf50"
        elif a <= hi: return "#ffeb3b"
        return "#f44336"

    if mission_data:
        n_cols = 2 + 2 * len(all_missions)   # Model + Cd Bound + 2 per mission
        html_content.append("    <div class='validation-card'>")
        html_content.append("        <div class='validation-title'>Model Performance Summary – Reentry Error by Mission</div>")
        html_content.append("        <table>")
        html_content.append("            <thead>")
        # Row 1: Model | Cd Bound | MISSION (colspan=2) ...
        html_content.append("            <tr>")
        html_content.append("                <th rowspan='2' style='vertical-align:middle'>Atmospheric Model</th>")
        html_content.append("                <th rowspan='2' style='vertical-align:middle'>Cd Bound</th>")
        for mission in all_missions:
            html_content.append(
                f"                <th colspan='2' style='text-align:center;"
                f"border-bottom:1px solid #555'>{mission}</th>")
        html_content.append("            </tr>")
        # Row 2: sub-headers
        html_content.append("            <tr>")
        for _ in all_missions:
            html_content.append("                <th>Err (days)</th>")
            html_content.append("                <th>Err (%)</th>")
        html_content.append("            </tr>")
        html_content.append("            </thead>")
        html_content.append("            <tbody>")

        prev_model = None
        for model in MODEL_ORDER:
            for bound in ["Cd Min", "Cd Max"]:
                key = (model, bound)
                if key not in mission_data:
                    continue
                # Visual separator between model groups
                if model != prev_model and prev_model is not None:
                    html_content.append(
                        f"            <tr><td colspan='{n_cols}' "
                        "style='border-top:2px solid #444;padding:0'></td></tr>")
                prev_model = model

                html_content.append("            <tr>")
                html_content.append(f"                <td><strong>{model}</strong></td>")
                html_content.append(f"                <td>{bound}</td>")
                for mission in all_missions:
                    m_data = mission_data[key].get(mission)
                    if m_data:
                        c = _color(m_data["err_pct"])
                        html_content.append(f"                <td>{m_data['err_days']:+.2f}</td>")
                        html_content.append(
                            f"                <td style='color:{c};font-weight:bold'>"
                            f"{m_data['err_pct']:+.2f}%</td>")
                    else:
                        html_content.append("                <td>&#8212;</td>")
                        html_content.append("                <td>&#8212;</td>")
                html_content.append("            </tr>")

        html_content.append("            </tbody>")
        html_content.append("        </table>")
        html_content.append("    </div>")


    for csv_file in sorted(csv_files):
        filename = os.path.basename(csv_file)
        basename = filename.replace("_validation.csv", "")
        
        # Parse CSV
        headers = []
        rows = []
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)
                rows = list(reader)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
            
        # Extract mission name and date from basename (e.g. grace1_20020317_20180309)
        parts = basename.split("_")
        if len(parts) >= 2:
            mission = parts[0].upper()
            date_str = parts[1]
            if len(date_str) == 8:
                formatted_date = f"{date_str[6:8]}-{date_str[4:6]}-{date_str[0:4]}"
            else:
                formatted_date = date_str
                
            if len(parts) >= 3:
                end_date_str = parts[2]
                if len(end_date_str) == 8:
                    formatted_end = f"{end_date_str[6:8]}-{end_date_str[4:6]}-{end_date_str[0:4]}"
                else:
                    formatted_end = end_date_str
                title_str = f"{mission} (Initial Date: {formatted_date} | End Date: {formatted_end})"
            else:
                title_str = f"{mission} (Initial Date: {formatted_date})"
        else:
            title_str = f"Mission TLE: {basename}"
            
        html_content.append("    <div class='validation-card'>")
        html_content.append(f"        <div class='validation-title'>{title_str}</div>")
        
        # Table
        html_content.append("        <table>")
        html_content.append("            <thead><tr>")
        for h in headers:
            html_content.append(f"                <th>{h}</th>")
        html_content.append("            </tr></thead>")
        html_content.append("            <tbody>")
        # Indices for percentage columns
        err_pct_idx = headers.index("Err (%)") if "Err (%)" in headers else -1
        alt_err_pct_idx = headers.index("Alt Err (%)") if "Alt Err (%)" in headers else -1

        for r in rows:
            html_content.append("            <tr>")
            for i, cell in enumerate(r):
                color_style = ""
                if i in (err_pct_idx, alt_err_pct_idx):
                    try:
                        val = abs(float(cell))
                        if val <= 5.0:
                            color_style = " style='color: #4caf50; font-weight: bold;'" # Green
                        elif val <= 20.0:
                            color_style = " style='color: #ffeb3b; font-weight: bold;'" # Yellow
                        else:
                            color_style = " style='color: #f44336; font-weight: bold;'" # Red
                    except ValueError:
                        pass
                html_content.append(f"                <td{color_style}>{cell}</td>")
            html_content.append("            </tr>")
        html_content.append("            </tbody>")
        html_content.append("        </table>")
        
        # Images (Look for single model or all models images)
        html_content.append("        <div class='image-gallery'>")
        
        # Potential image patterns based on the validation script:
        # Single model: {basename}_{model_name}_evolution.png
        # All models: {basename}_all_evolution.png
        image_pattern = os.path.join(data_dir, f"{basename}*.png")
        images = sorted(glob.glob(image_pattern))
        
        for img_path in images:
            img_filename = os.path.basename(img_path)
            # Relative path from the results directory
            rel_img_path = f"results_decay_validations/{img_filename}"
            html_content.append("            <div class='image-card'>")
            html_content.append(f"                <img src='{rel_img_path}' alt='{img_filename}'>")
            html_content.append("            </div>")
            
        html_content.append("        </div>")
        html_content.append("    </div>")

    # Footer
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_content.append(f"    <div class='timestamp'>Report Generated: {now_str}</div>")
    html_content.append("</body>")
    html_content.append("</html>")

    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(html_content))
        print(f"Successfully generated report at: {report_path}")
    except Exception as e:
        print(f"Failed to write report: {e}")

if __name__ == "__main__":
    generate_report()
