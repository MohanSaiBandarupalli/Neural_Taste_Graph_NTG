"""API load test: python scripts/api_load_test.py > reports/v4/synthetic/api_load_test.json (API must be running on :8000)."""
import httpx, time, json
from concurrent.futures import ThreadPoolExecutor
B="http://127.0.0.1:8000"
C=httpx.Client(timeout=30, limits=httpx.Limits(max_keepalive_connections=16))
t=httpx.post(B+"/v1/auth/login",json={"email":"viewer@demo.ntg","password":"demo-password-123"}).json()["token"]
H={"Authorization":f"Bearer {t}"}
pids=[p["id"] for p in httpx.get(B+"/v1/profiles",headers=H).json()]
def one(i):
    t0=time.perf_counter(); r=C.post(f"{B}/v1/profiles/{pids[i%3]}/recommendations",headers=H,json={"k":20,"explain":i%2==0},timeout=30); return (time.perf_counter()-t0)*1000, r.status_code
out={"endpoint":"POST /v1/profiles/{id}/recommendations (k=20, 50% with explanations)","server":"1 uvicorn worker, in-process engine","client":"httpx keep-alive pool","evidence_type":"observed"}
for conc in (1,4,8):
    n=200; t0=time.perf_counter()
    with ThreadPoolExecutor(conc) as ex: res=list(ex.map(one,range(n)))
    el=time.perf_counter()-t0; lat=sorted(x[0] for x in res)
    out[f"concurrency_{conc}"]={"n":n,"rps":round(n/el,1),"p50_ms":round(lat[n//2],1),"p95_ms":round(lat[int(n*.95)],1),"p99_ms":round(lat[int(n*.99)],1),"errors":sum(1 for x in res if x[1]!=200)}
print(json.dumps(out, indent=1))
