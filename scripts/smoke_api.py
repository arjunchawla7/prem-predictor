import json
import urllib.request

def get(u):
    return json.loads(urllib.request.urlopen(u, timeout=60).read())

d = get("http://127.0.0.1:5000/api/gameweek/1")
f = d["fixtures"][2]   # Everton v Crystal Palace (full data)
print("fixture:", f["home"]["name"], "v", f["away"]["name"], "| mode:", f["lineup_mode"])
p = f["prediction"]
print("tag:", p["tag"], "| probs:", p["p_home"], p["p_draw"], p["p_away"],
      "| top:", p["top_scores"][:2])
print("home XI first 3:", [(x["name"], x["position"], x["tier"], x["fatigue"])
                           for x in (p["home_lineup"] or [])[:3]])

squad = get(f"http://127.0.0.1:5000/api/squad/{f['home']['id']}")
print("squad size:", len(squad), "| top:", [(s["name"], s["minutes"]) for s in squad[:3]])

# manual prediction round-trip
import urllib.request as ur
xi = [s["id"] for s in squad[:11]]
req = ur.Request(f"http://127.0.0.1:5000/api/fixture/{f['id']}/manual",
                 data=json.dumps({"home": xi, "away": None}).encode(),
                 headers={"Content-Type": "application/json"}, method="POST")
r = json.loads(ur.urlopen(req, timeout=120).read())
print("manual:", r["ok"], r["prediction"]["p_home"], r["prediction"]["notes"][:80])

d2 = get("http://127.0.0.1:5000/api/gameweek/1")
f2 = [x for x in d2["fixtures"] if x["id"] == f["id"]][0]
print("mode after manual:", f2["lineup_mode"], "| tag:", f2["prediction"]["tag"])

req = ur.Request(f"http://127.0.0.1:5000/api/fixture/{f['id']}/reset",
                 data=b"{}", headers={"Content-Type": "application/json"},
                 method="POST")
print("reset:", json.loads(ur.urlopen(req, timeout=30).read()))
d3 = get("http://127.0.0.1:5000/api/gameweek/1")
f3 = [x for x in d3["fixtures"] if x["id"] == f["id"]][0]
print("mode after reset:", f3["lineup_mode"], "| tag:", f3["prediction"]["tag"])
