#!/usr/bin/env python3
"""Build encounter_geometry.npz (analytic: Horizons Apophis + Meeus Sun, J2000 equatorial geocentric)
and apophis_set3_cov.npz (anchored sol-220 B-plane cov6) into uq_montecarlo_v2/data/.  Pure numpy."""
import os, re, numpy as np, datetime as dt

ANIM = "/sessions/tender-fervent-darwin/mnt/Animation"
V2DATA = os.path.join(ANIM, "Designs/D_strobe_fullbody/uq_montecarlo_v2/data")
HOR = os.path.join(ANIM, "data/ephemeris/Apophis_horizons_results.txt")
AU_M = 1.495978707e11; DEG = np.pi/180

# ---- CA epoch (UTC) -> JD(TDB) ----
ca = dt.datetime(2029, 4, 13, 20, 11, 21, 903000, tzinfo=dt.timezone.utc)
def jd_utc(d):
    a=(14-d.month)//12; y=d.year+4800-a; m=d.month+12*a-3
    jdn=d.day+(153*m+2)//5+365*y+y//4-y//100+y//400-32045
    return (jdn-0.5)+(d.hour*3600+d.minute*60+d.second+d.microsecond/1e6)/86400.0
JD_TDB = jd_utc(ca) + 69.184/86400.0     # UTC->TT(~TDB) in 2029 (TAI-UTC=37, TT-TAI=32.184)
print(f"CA epoch {ca.isoformat()}  -> JD(TDB) {JD_TDB:.6f}")

# ---- parse Horizons geocentric ICRF (J2000) state, interp to CA ----
lines = open(HOR, errors="replace").read().splitlines()
i = next(k for k,l in enumerate(lines) if l.strip()=="$$SOE")+1
numre = re.compile(r"[-+]?\d+\.\d+E[+-]\d+")
jd, pos, vel = [], [], []
while i < len(lines):
    l=lines[i].strip()
    if l.startswith("$$EO") or not l: break
    jd.append(float(l.split()[0]))
    pos.append([float(x) for x in numre.findall(lines[i+1])][:3])
    vel.append([float(x) for x in numre.findall(lines[i+2])][:3]); i+=3
jd=np.array(jd); pos=np.array(pos)*1e3; vel=np.array(vel)*1e3   # km->m, km/s->m/s
o=np.argsort(jd); jd,pos,vel=jd[o],pos[o],vel[o]
r_apo=np.array([np.interp(JD_TDB,jd,pos[:,k]) for k in range(3)])
v_apo=np.array([np.interp(JD_TDB,jd,vel[:,k]) for k in range(3)])
print(f"Apophis geocentric @CA: |r|={np.linalg.norm(r_apo)/1e3:.0f} km  |v|={np.linalg.norm(v_apo)/1e3:.3f} km/s")

# ---- Meeus low-precision Sun (geocentric, J2000 equatorial) ----
T=(JD_TDB-2451545.0)/36525.0
L0=(280.46646+36000.76983*T+0.0003032*T*T)%360
M=(357.52911+35999.05029*T-0.0001537*T*T)%360; Mr=M*DEG
C=((1.914602-0.004817*T-1.4e-5*T*T)*np.sin(Mr)+(0.019993-0.000101*T)*np.sin(2*Mr)+0.000289*np.sin(3*Mr))
lam=(L0+C)*DEG                                   # true ecliptic longitude
e=0.016708634-0.000042037*T; nu=Mr+C*DEG
R=(1.000001018*(1-e*e)/(1+e*np.cos(nu)))*AU_M    # Sun distance [m]
eps=(23.439291-0.0130042*T)*DEG                  # obliquity
r_sun=R*np.array([np.cos(lam), np.sin(lam)*np.cos(eps), np.sin(lam)*np.sin(eps)])  # Earth->Sun, equ J2000

# ---- unit vectors (Apophis-centred), phase, implied ecliptic longitude ----
sun_vec=r_sun-r_apo; sun_hat=sun_vec/np.linalg.norm(sun_vec)     # Apophis->Sun
earth_hat=-r_apo/np.linalg.norm(r_apo)                          # Apophis->Earth
phase=np.degrees(np.arccos(np.clip(sun_hat@earth_hat,-1,1)))
# implied ecliptic longitude of sun_hat (equ->ecl)
ye=sun_hat[1]*np.cos(eps)+sun_hat[2]*np.sin(eps); xe=sun_hat[0]
sun_ecl_lon=np.degrees(np.arctan2(ye,xe))%360
print(f"\nSun:  ecliptic longitude {sun_ecl_lon:.2f} deg   (V1/V2 default was 203 deg)")
print(f"      sun_hat (J2000 equ) = {np.round(sun_hat,4)}")
print(f"      earth_hat           = {np.round(earth_hat,4)}")
print(f"      Sun-Apophis-Earth phase angle = {phase:.2f} deg   Apophis-Sun dist {np.linalg.norm(sun_vec)/AU_M:.4f} AU")
d=(sun_ecl_lon-203)%360; d=min(d,360-d)
print(f"      >>> real vs 203deg default differ by {d:.1f} deg  {'(~180deg -> default points ANTI-sun; lit/dark would be FLIPPED)' if d>150 else ''}")

os.makedirs(V2DATA, exist_ok=True)
np.savez(os.path.join(V2DATA,"encounter_geometry.npz"),
         sun_hat=sun_hat, earth_hat=earth_hat, r_sun_geo_m=r_sun, r_apo_geo_m=r_apo,
         sun_apophis_dist_m=np.linalg.norm(sun_vec), phase_deg=phase,
         sun_ecliptic_lon_deg=sun_ecl_lon, jd_tdb=JD_TDB,
         source=np.array(["ANALYTIC: Horizons Apophis + Meeus Sun, J2000 equ geocentric. "
                          "Verify/replace with compute_encounter_geometry.py (SPICE) in the Basilisk env."]))
print("\nwrote encounter_geometry.npz")

# ---- anchored sol-220 B-plane cov6 (along-track dominant, geocentric m / m/s) ----
ahat=v_apo/np.linalg.norm(v_apo)                 # along-track ~ relative-velocity direction at CA
sig_along, sig_perp, sig_v = 735.0, 300.0, 5.0e-4    # Farnocchia sol-220: sub-km B-plane, ~0.5 mm/s
P=np.outer(ahat,ahat); Q=np.eye(3)-P
pos_cov=sig_along**2*P+sig_perp**2*Q
cov6=np.zeros((6,6)); cov6[:3,:3]=pos_cov; cov6[3:,3:]=(sig_v**2)*np.eye(3)
w=np.linalg.eigvalsh(cov6)
print(f"\ncov6 PSD: min eig {w.min():.3e} (>=0)  pos 1-sig total {np.sqrt(np.trace(pos_cov)):.0f} m  vel 1-sig {sig_v*1e3:.2f} mm/s")
np.savez(os.path.join(V2DATA,"apophis_set3_cov.npz"), cov6=cov6,
         note=np.array(["anchored sol-220 B-plane: along-track-dominant (sig_along=735, sig_perp=300 m; "
                        "sig_v=0.5 mm/s), along-track = Apophis geocentric velocity dir at CA. "
                        "Exact element->state mapping needs sbp-99942-220 partials."]))
print("wrote apophis_set3_cov.npz")
