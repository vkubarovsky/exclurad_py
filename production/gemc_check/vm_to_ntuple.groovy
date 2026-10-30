// exclurad_py VECTOR MESON CLAS12 MC -> flat binary ntuple (little-endian float32).
//   run-groovy vm_to_ntuple.groovy <file.hipo> <max events> <out.f32> [skip] [channel]
// channel = jpsi | phi | rho   (default jpsi).  Replaces jpsi_to_ntuple.groovy,
// which was hard-wired to pid 443 decaying to e+e-.  On a J/psi file this script
// writes a BYTE-IDENTICAL record to the one jpsi_to_ntuple.groovy writes; that is
// checked, not assumed (see RESULTS in docs).
// Same calling convention as hipo_to_ntuple.groovy, so sim_array.sbatch can run it
// unchanged (copy it to $PROD_DIR/hipo_to_ntuple.groovy).
//
//   channel   meson pid   daughters        note
//   jpsi        443       e+ e-  (-11/+11) decay-lepton radiation possible (idecay 2)
//   phi         333       K+ K-  (321/-321)
//   rho         113       pi+ pi- (211/-211)
//
// Particles, identified through MC::Lund (pid + parent column, which survive GEMC)
// and matched to MC::Particle by momentum, since MC::Particle drops the status-0
// vector-meson row and the row indices do not line up:
//   k=0 e'      pid 11,  parent 0          (scattered electron)
//   k=1 p       pid = hadron pid (2212)
//   k=2 h+      POSITIVE daughter, parent = meson row
//   k=3 h-      NEGATIVE daughter, parent = meson row
//   k=4 gdec    pid 22,  parent = meson row (decay radiation, J/psi idecay 2; may be absent)
//   k=5 gprod   pid 22,  parent 0          (production radiation; may be absent)
// Record per event (NFLD = 80 floats):
//   [0]       weight (MC::Event)
//   [1..18]   gen px,py,pz for k = 0..5 (zeros if the particle is absent)
//   [19..66]  for k = 0..5, block of 8 starting at 19+8k:
//             found(0/1) pid charge status quality px py pz -- REC::Particle row given by MC::GenMatch
//   [67] n_rec(negative daughter pid) [68] n_rec(positive daughter pid)
//   [69] n_hadron [70] n_gamma [71] n_rec rows
//   [72] bit mask of generated particles present (1<<k)
import org.jlab.jnp.hipo4.data.*
import org.jlab.jnp.hipo4.io.*
import java.nio.*

def fname = args[0]
int maxev = args[1] as int
def oname = args[2]
int skip = args.size() > 3 ? args[3] as int : 0
def chan = args.size() > 4 ? args[4].toLowerCase() : "jpsi"
// Daughter pids per channel.  dplus is the POSITIVE track (slot 2), dminus the
// NEGATIVE one (slot 3) -- for e+e- that means pid -11 and +11, because the
// lepton pid sign is opposite to the charge.  Getting this backwards would
// silently swap two slots, so it is written out per channel, not derived.
int mpid, dplus, dminus
if      (chan == "jpsi") { mpid = 443; dplus = -11;  dminus =  11  }
else if (chan == "phi" ) { mpid = 333; dplus = 321;  dminus = -321 }
else if (chan == "rho" ) { mpid = 113; dplus = 211;  dminus = -211 }
else throw new IllegalArgumentException("vm_to_ntuple: unknown channel '" + chan + "' (jpsi|phi|rho)")
int hpid = 2212
final int NFLD = 80
final int NP = 6

def reader = new HipoReader(); reader.open(fname)
def ev = new Event()
def lund = reader.getBank("MC::Lund"), mcp = reader.getBank("MC::Particle"), mce = reader.getBank("MC::Event")
def gm = reader.getBank("MC::GenMatch"), rec = reader.getBank("REC::Particle")
def os = new BufferedOutputStream(new FileOutputStream(oname), 1<<20)
def bb = ByteBuffer.allocate(NFLD*4).order(ByteOrder.LITTLE_ENDIAN)
// THE SELF-DESCRIBING HEADER, 64 bytes, in front of the records.
// Written since 2026-10-23 because a bare float32 stream cannot say which
// vintage it is: 1600 superseded ntuples, named exactly like the current ones
// but 73 floats where the current record is 80, were read as current by a
// library pinned to the old length and a whole analysis came out of them with
// no error raised.  Layout and the reasoning are in
// production/analysis/f32_format.py, which reads it; keep the two in step.
//   0  'EXF3'   4  version=1   8  nfields   12  nrecords(0=unknown)
//  16  header_bytes=64         20  channel[16]      36  commit[16]
//  52  reserved[12]
// nrecords is written as 0 here: the count is not known until the loop ends
// and this is a streaming writer.  f32_format derives it from the file size
// and only cross-checks when the field is non-zero.
def writeHeader = { out, nfld, chanName, gitSha ->
  def h = ByteBuffer.allocate(64).order(ByteOrder.LITTLE_ENDIAN)
  h.put("EXF3".getBytes("US-ASCII"))
  h.putInt(1); h.putInt(nfld); h.putInt(0); h.putInt(64)
  byte[] c = new byte[16]; byte[] cb = chanName.getBytes("US-ASCII")
  System.arraycopy(cb, 0, c, 0, Math.min(cb.length, 16)); h.position(20); h.put(c)
  byte[] g = new byte[16]; byte[] gb = (gitSha ?: "").getBytes("US-ASCII")
  System.arraycopy(gb, 0, g, 0, Math.min(gb.length, 16)); h.position(36); h.put(g)
  out.write(h.array())
}
def gitCommit = {
  try {
    def p = "git -C ${System.getenv('EXRU') ?: System.getProperty('user.home') + '/exclurad_py'} rev-parse --short HEAD".execute()
    p.waitFor(); return p.exitValue() == 0 ? p.text.trim() : ""
  } catch (Throwable t) { return "" }
}()

writeHeader(os, NFLD, chan, gitCommit)
float[] r = new float[NFLD]

int n = 0, nw = 0, nbad = 0, nskip = 0
int[] gi = new int[NP]
double[][] gp = new double[NP][3]
long t0 = System.currentTimeMillis()
try {
  while (reader.hasNext() && nskip < skip) { reader.nextEvent(ev); nskip++ }
  while (reader.hasNext() && n < maxev) {
    reader.nextEvent(ev); n++
    ev.read(lund); ev.read(mcp); ev.read(mce); ev.read(gm); ev.read(rec)
    Arrays.fill(r, 0f); Arrays.fill(gi, -1)
    r[0] = mce.getRows() > 0 ? mce.getFloat("weight", 0) : 1f

    // 1. classify the LUND rows (1-based parent index)
    int psirow = -1
    for (int i = 0; i < lund.getRows(); i++) if (lund.getInt("pid", i) == mpid) { psirow = i + 1; break }
    boolean[] have = new boolean[NP]
    for (int i = 0; i < lund.getRows(); i++) {
      int pid = lund.getInt("pid", i), par = lund.getInt("parent", i)
      int k = -1
      if (pid == 11 && par == 0) k = 0
      else if (pid == hpid) k = 1
      else if (pid == dplus  && par == psirow) k = 2
      else if (pid == dminus && par == psirow) k = 3
      else if (pid == 22 && par == psirow) k = 4
      else if (pid == 22 && par == 0) k = 5
      if (k < 0 || have[k]) continue
      have[k] = true
      gp[k][0] = lund.getFloat("px", i); gp[k][1] = lund.getFloat("py", i); gp[k][2] = lund.getFloat("pz", i)
      if (k == 0) r[73] = lund.getFloat("vz", i)   // one vertex per event; take it from e'
    }
    if (!(have[0] && have[1] && have[2] && have[3])) { nbad++; continue }

    // 2. find each in MC::Particle by momentum (GenMatch refers to MC::Particle rows)
    def src = mcp.getRows() > 0 ? mcp : lund
    for (int k = 0; k < NP; k++) {
      if (!have[k]) continue
      for (int i = 0; i < src.getRows(); i++) {
        if (Math.abs(src.getFloat("px", i) - gp[k][0]) < 1e-5 && Math.abs(src.getFloat("py", i) - gp[k][1]) < 1e-5
            && Math.abs(src.getFloat("pz", i) - gp[k][2]) < 1e-5) { gi[k] = i; break }
      }
      r[1+3*k] = (float) gp[k][0]; r[2+3*k] = (float) gp[k][1]; r[3+3*k] = (float) gp[k][2]
      r[72] += (1 << k)
    }

    // 3. reconstructed partner of each generated particle
    for (int j = 0; j < gm.getRows(); j++) {
      int mci = gm.getShort("mcindex", j), pi = gm.getShort("pindex", j)
      if (pi < 0 || pi >= rec.getRows()) continue
      for (int k = 0; k < NP; k++) {
        if (gi[k] < 0 || mci != gi[k]) continue
        int b = 19 + 8*k
        r[b] = 1f; r[b+1] = rec.getInt("pid", pi); r[b+2] = rec.getByte("charge", pi); r[b+3] = rec.getShort("status", pi)
        r[b+4] = gm.getFloat("quality", j)
        r[b+5] = rec.getFloat("px", pi); r[b+6] = rec.getFloat("py", pi); r[b+7] = rec.getFloat("pz", pi)
        r[74 + k] = rec.getFloat("vz", pi)
      }
    }
    int nem=0, nep=0, nh=0, ng=0
    for (int i = 0; i < rec.getRows(); i++) {
      int pid = rec.getInt("pid", i)
      if (pid == dminus) nem++; else if (pid == dplus) nep++; else if (pid == hpid) nh++; else if (pid == 22) ng++
    }
    r[67]=nem; r[68]=nep; r[69]=nh; r[70]=ng; r[71]=rec.getRows()
    bb.clear(); for (int k = 0; k < NFLD; k++) bb.putFloat(r[k])
    os.write(bb.array()); nw++
  }
} catch (Throwable e) {
  System.err.println("FATAL at event $n : " + e); e.printStackTrace(); os.close(); System.exit(2)
}
os.close()
double dt=(System.currentTimeMillis()-t0)/1000.0
println("DONE skipped $nskip read $n written $nw badGen $nbad  ${(int)(n/Math.max(dt,1e-3))} ev/s -> $oname")
