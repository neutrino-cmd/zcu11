import numpy as np
from qick import QickSoc, AveragerProgram
import time

RO_CH = 0                 # ADC224_T0_CH0
SIG_MHZ = 4760.0          
FS_ADC = 4096.0           # ADC sample rate, MHz
ALIAS_MHZ = SIG_MHZ % FS_ADC          # 664.0, where it actually lands
DEC_FS = 512e6            # decimated rate
ADC_BITS = 12
ADC_FULLSCALE_VPP = 1.0   # Modify after counts-volt calibration

LSB_V = ADC_FULLSCALE_VPP / 2 ** ADC_BITS
COUNTS_FULLSCALE = 2 ** (ADC_BITS - 1)

class TwoChannelExternal(AveragerProgram):
    def initialize(self):
        cfg = self.cfg
        for ch in cfg["ro_chs"]:
            self.declare_readout(ch=ch, length=cfg["ro_len"],
                                 freq=cfg["ddc_mhz"], gen_ch=None)
        self.synci(200)

    def body(self):
        cfg = self.cfg
        self.trigger(adcs=cfg["ro_chs"], pins=[0],
                     adc_trig_offset=cfg["adc_trig_offset"])
        self.wait_all()
        self.sync_all(self.us2cycles(cfg["relax_delay"]))


def capture_two_ch_external(soc, soccfg, ddc_mhz=200.0, ro_len=1000,
                            adc_trig_offset=0, relax_delay=1.0, timeout=60):
    cfg = {"ro_chs": [0, 1], "ddc_mhz": ddc_mhz, "ro_len": ro_len,
           "adc_trig_offset": adc_trig_offset, "relax_delay": relax_delay,
           "reps": 1, "soft_avgs": 1}
    prog = TwoChannelExternal(soccfg, cfg)
    prog.config_all(soc)
    prog.config_bufs(soc, enable_avg=True, enable_buf=True)

    soc.start_src("external")
    soc.clear_tproc_counter(addr=prog.counter_addr)
    t0 = time.time()
    while soc.get_tproc_counter(addr=prog.counter_addr) < 1:
        if time.time() - t0 > timeout:
            soc.start_src("internal")
            raise TimeoutError("no trigger in %.0f s" % timeout)
        time.sleep(0.001)
    waited = time.time() - t0

    d0 = np.asarray(soc.get_decimated(ch=0, address=0, length=ro_len))
    d1 = np.asarray(soc.get_decimated(ch=1, address=0, length=ro_len))
    z0 = d0[:, 0] + 1j * d0[:, 1]
    z1 = d1[:, 0] + 1j * d1[:, 1]
    t = np.arange(len(z0)) / DEC_FS * 1e6

    soc.start_src("internal")
    print("triggered after %.2f s" % waited)
    return t, z0, z1

t, z0, z1 = capture_two_ch_external(soc, soccfg, ddc_mhz=200.0)

print("ch0 amplitude:", np.abs(z0).mean())
print("ch1 amplitude:", np.abs(z1).mean())

d = z0 * np.conj(z1)
core = slice(len(d)//4, None)
phases = np.angle(d[core])

mean_phase = np.rad2deg(np.angle(np.mean(np.exp(1j*phases))))
v = np.mean(np.exp(1j*phases))
spread = np.rad2deg(np.sqrt(-2*np.log(np.abs(v))))
np.savez("tmr_shot_%s.npz" % time.strftime("%Y%m%d_%H%M%S"),
        t=t, z0=z0, z1=z1, mean_deg=mean_phase)
print("phase difference: %.3f deg" % mean_phase)

