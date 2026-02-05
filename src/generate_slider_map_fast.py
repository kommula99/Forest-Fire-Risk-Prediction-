import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "risk_7days.csv"
OUT = ROOT / "maps" / "slider_map.html"

df = pd.read_csv(DATA)

points = []
for _, r in df.iterrows():
    points.append({
        "lat": r["lat"],
        "lon": r["lon"],
        "risk": [r[f"day_{d}"] for d in range(1,8)]
    })

html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>7-Day Fire Risk Slider</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
</head>
<body>

<h3 style="text-align:center">🔥 Forest Fire Risk Prediction (Next 7 Days)</h3>
<div style="text-align:center">
<input type="range" min="1" max="7" value="1" id="daySlider" />
<span id="dayLabel">Day 1</span>
</div>

<div id="map" style="height:600px"></div>

<script>
var map = L.map('map').setView([20, 80], 4);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png').addTo(map);

var data = {json.dumps(points)};
var markers = [];

function color(p) {{
  if (p > 0.66) return "red";
  if (p > 0.33) return "orange";
  return "green";
}}

function render(day) {{
  markers.forEach(m => map.removeLayer(m));
  markers = [];
  data.forEach(p => {{
    let c = L.circleMarker([p.lat, p.lon], {{
      radius: 5,
      color: color(p.risk[day-1]),
      fillOpacity: 0.7
    }}).addTo(map);
    markers.push(c);
  }});
}}

document.getElementById("daySlider").oninput = function() {{
  document.getElementById("dayLabel").innerText = "Day " + this.value;
  render(parseInt(this.value));
}}

render(1);
</script>
</body>
</html>
"""

OUT.write_text(html)
print("Slider map created:", OUT)
