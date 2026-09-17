import numpy as np
import matplotlib.pyplot as plt
from qick import QickSoc, AveragerProgram

RO_CH = 0                 # ADC224_T0_CH0
SIG_MHZ = 4760.0          
FS_ADC = 4096.0           # ADC sample rate, MHz
ALIAS_MHZ = SIG_MHZ % FS_ADC          # 664.0, where it actually lands
DEC_FS = 512e6            # decimated rate
ADC_BITS = 12
ADC_FULLSCALE_VPP = 1.0   # Modify after counts-volt calibration

LSB_V = ADC_FULLSCALE_VPP / 2 ** ADC_BITS
COUNTS_FULLSCALE = 2 ** (ADC_BITS - 1)


class TriggeredCapture(AveragerProgram):

    def initialize(self):
        cfg = self.cfg
        self.declare_readout(ch=cfg["ro_ch"], length=cfg["ro_len"],
                             freq=cfg["ddc_mhz"], gen_ch=None)
        self.synci(200)

    def body(self):
        cfg = self.cfg
        self.trigger(adcs=[cfg["ro_ch"]],
                     mr=cfg.get("use_mr", False),
                     pins=[0],
                     adc_trig_offset=cfg["adc_trig_offset"])
        self.wait_all()
        self.sync_all(self.us2cycles(cfg["relax_delay"]))


def arm_external(soc):
    soc.start_src("external")
    print("tProc armed for external start on PMOD1_0_LS")


def arm_internal(soc):
    soc.start_src("internal")


def capture_one(soc, soccfg, ddc_mhz=ALIAS_MHZ, ro_len=200,
                adc_trig_offset=0, relax_delay=1.0, use_mr=False, timeout=None):
    cfg = {"ro_ch": RO_CH, "ddc_mhz": ddc_mhz, "ro_len": ro_len,
           "adc_trig_offset": adc_trig_offset, "relax_delay": relax_delay,
           "use_mr": use_mr, "reps": 1, "soft_avgs": 1}
    if use_mr:
        soc.arm_mr(ch=RO_CH)
    prog = TriggeredCapture(soccfg, cfg)
    iq = prog.acquire_decimated(soc, progress=False)
    I, Q = iq[0]
    z = I + 1j * Q
    t = np.arange(len(z)) / DEC_FS * 1e6
    if use_mr:
        mr = np.asarray(soc.get_mr())
        raw = mr[:, 0].astype(float) + 1j * mr[:, 1].astype(float)
        return t, z, raw
    return t, z

import time

def capture_external(soc, soccfg, ddc_mhz=ALIAS_MHZ, ro_len=200, buf="avg",
                     adc_trig_offset=0, relax_delay=1.0, timeout=60):

    cfg = {"ro_ch": RO_CH, "ddc_mhz": ddc_mhz, "ro_len": ro_len,
           "adc_trig_offset": adc_trig_offset, "relax_delay": relax_delay,
           "use_mr": True, "reps": 1, "soft_avgs": 1}
    prog = TriggeredCapture(soccfg, cfg)
    prog.config_all(soc)
    prog.config_bufs(soc, enable_avg=True, enable_buf=True)
    if buf == "mr":
        soc.arm_mr(ch=RO_CH)

    soc.start_src("external")
    soc.clear_tproc_counter(addr=prog.counter_addr)
    t0 = time.time()
    while soc.get_tproc_counter(addr=prog.counter_addr) < 1:
        if time.time() - t0 > timeout:
            soc.start_src("internal")
            raise TimeoutError("no trigger in %.0f s" % timeout)
        time.sleep(0.001)
    waited = time.time() - t0
    soc.start_src("internal")
    print("triggered after %.2f s" % waited)

    if buf == "mr":
        mr = np.asarray(soc.get_mr())
        z = mr[:, 0].astype(float) + 1j * mr[:, 1].astype(float)
        t = np.arange(len(z)) / FS_ADC / 1e6 * 1e6
        return t, z

    d = np.asarray(soc.get_decimated(ch=RO_CH, address=0, length=ro_len))
    I, Q = d[:, 0], d[:, 1]
    z = I + 1j * Q
    t = np.arange(len(z)) / DEC_FS * 1e6
    return t, z


def shot_parameters(t_us, z, thresh=0.3, settle_frac=0.2):

    amp = np.abs(z)
    if amp.max() < 1e-9:
        return None
    on = amp > thresh * amp.max()
    if on.sum() < 3:
        return None
    i0, i1 = np.where(on)[0][[0, -1]]
    n = i1 - i0 + 1
    g = max(1, int(settle_frac * n))
    core = slice(i0 + g, i1 - g + 1) if n > 2 * g + 2 else slice(i0, i1 + 1)

    zc = z[core]
    vec = zc.mean()
    return {
        "amplitude_counts": np.abs(zc).mean(),
        "phase_deg": np.rad2deg(np.angle(vec)),
        "amp_rms_pct": 100 * np.abs(zc).std() / np.abs(zc).mean(),
        "n_samples": len(zc),
    }


def acquire_n(soc, soccfg, n_shots, ddc_mhz=ALIAS_MHZ, ro_len=200,
              adc_trig_offset=0, verbose=True):

    rows = []
    for i in range(n_shots):
        t, z = capture_one(soc, soccfg, ddc_mhz, ro_len, adc_trig_offset)
        p = shot_parameters(t, z)
        if p is None:
            if verbose:
                print("  shot %3d: nothing above threshold" % i)
            continue
        rows.append(p)
        if verbose:
            print("  shot %3d: amp %8.1f counts (%.3f mV)  phase %+7.2f deg  "
                  "len %5.1f ns" % (i, p["amplitude_counts"],
                                    p["amplitude_volts"] * 1e3,
                                    p["phase_deg"], p["length_ns"]))
    if rows:
        a = np.array([r["amplitude_counts"] for r in rows])
        ph = np.array([r["phase_deg"] for r in rows])
        v = np.exp(1j * np.deg2rad(ph))
        spread = np.rad2deg(np.sqrt(-2 * np.log(np.abs(v.mean()))))
        print("\n%d shots: amplitude %.1f +/- %.1f counts (%.2f%%), "
              "phase spread %.2f deg" % (len(rows), a.mean(), a.std(),
                                         100 * a.std() / a.mean(), spread))
    return rows

def acquire_n_external(soc, soccfg, n_shots, ddc_mhz=ALIAS_MHZ, ro_len=200, buf="avg",
                       adc_trig_offset=0, verbose=True):
    rows = []
    for i in range(n_shots):
        t, z = capture_external(soc, soccfg, ddc_mhz, ro_len, buf, adc_trig_offset)
        p = shot_parameters(t, z)
        if p is None:
            if verbose:
                print("  shot %3d: nothing above threshold" % i)
            continue
        rows.append(p)
        if verbose:
            print("  shot %3d: amp %8.1f counts  phase %+7.2f deg  len %5.1f ns"
                  % (i, p["amplitude_counts"], p["phase_deg"], p["length_ns"]))
    return rows

#Change parameters below when needed
t, z = capture_external(soc, soccfg, ddc_mhz=200.0, ro_len=1000, buf="avg")
print(shot_parameters(t, z))
np.savez("waveform.npz", t=t, I=z.real, Q=z.imag)
rows = acquire_n_external(soc, soccfg, n_shots=5, ddc_mhz=200.0, ro_len=1000, buf="mr")
mr_amp = np.array([r["amplitude_counts"] for r in rows])
mr_phase = np.array([r["phase_deg"] for r in rows])
mr_length = np.array([r["length_ns"] for r in rows])
np.savez("mr_waveform.npz",
        t=t, I=z.real, Q=z.imag,
        mr_amplitude=mr_amp, mr_phase=mr_phase, mr_length=mr_length)
#d = np.load("waveform.npz")
#I, Q = d["I"], d["Q"]

