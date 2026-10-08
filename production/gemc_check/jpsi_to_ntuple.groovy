// exclurad_py J/psi CLAS12 MC -> flat binary ntuple (little-endian float32, fixed record).
//   run-groovy jpsi_to_ntuple.groovy <file.hipo> <max events> <out.f32> [skip] [hadron pid]
// Same calling convention as hipo_to_ntuple.groovy, so sim_array.sbatch can run it
// unchanged (copy it to $PROD_DIR/hipo_to_ntuple.groovy).  The pi0 script requires
// the pi0's two photons and writes NOTHING for a J/psi event.
//
// Particles, identified through MC::Lund (pid + parent column, which survive GEMC)
// and matched to MC::Particle by momentum, since MC::Particle drops the status-0
// J/psi row and the row indices do not line up:
//   k=0 e'      pid 11,  parent 0          (scattered electron; FT for this sample)
//   k=1 p       pid = hadron pid (2212)
//   k=2 e+      pid -11, parent = J/psi row
//   k=3 e-      pid 11,  parent = J/psi row (decay electron)
//   k=4 gdec    pid 22,  parent = J/psi row (decay-lepton radiation, idecay 2; may be absent)
//   k=5 gprod   pid 22,  parent 0          (production radiation; may be absent)
// Record per event (NFLD = 73 floats):
//   [0]       weight (MC::Event)
//   [1..18]   gen px,py,pz for k = 0..5 (zeros if the particle is absent)
//   [19..66]  for k = 0..5, block of 8 starting at 19+8k:
//             found(0/1) pid charge status quality px py pz -- REC::Particle row given by MC::GenMatch
//   [67] n_e-(rec pid 11) [68] n_e+(rec pid -11) [69] n_hadron [70] n_gamma [71] n_rec rows
//   [72] bit mask of generated particles present (1<<k)
import org.jlab.jnp.hipo4.data.*
import org.jlab.jnp.hipo4.io.*
import java.nio.*

def fname = args[0]
int maxev = args[1] as int
def oname = args[2]
int skip = args.size() > 3 ? args[3] as int : 0
int hpid = args.size() > 4 ? args[4] as int : 2212
final int NFLD = 73
final int NP = 6

def reader = new HipoReader(); reader.open(fname)
def ev = new Event()
def lund = reader.getBank("MC::Lund"), mcp = reader.getBank("MC::Particle"), mce = reader.getBank("MC::Event")
def gm = reader.getBank("MC::GenMatch"), rec = reader.getBank("REC::Particle")
def os = new BufferedOutputStream(new FileOutputStream(oname), 1<<20)
def bb = ByteBuffer.allocate(NFLD*4).order(ByteOrder.LITTLE_ENDIAN)
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
    for (int i = 0; i < lund.getRows(); i++) if (lund.getInt("pid", i) == 443) { psirow = i + 1; break }
    boolean[] have = new boolean[NP]
    for (int i = 0; i < lund.getRows(); i++) {
      int pid = lund.getInt("pid", i), par = lund.getInt("parent", i)
      int k = -1
      if (pid == 11 && par == 0) k = 0
      else if (pid == hpid) k = 1
      else if (pid == -11 && par == psirow) k = 2
      else if (pid == 11 && par == psirow) k = 3
      else if (pid == 22 && par == psirow) k = 4
      else if (pid == 22 && par == 0) k = 5
      if (k < 0 || have[k]) continue
      have[k] = true
      gp[k][0] = lund.getFloat("px", i); gp[k][1] = lund.getFloat("py", i); gp[k][2] = lund.getFloat("pz", i)
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
      }
    }
    int nem=0, nep=0, nh=0, ng=0
    for (int i = 0; i < rec.getRows(); i++) {
      int pid = rec.getInt("pid", i)
      if (pid == 11) nem++; else if (pid == -11) nep++; else if (pid == hpid) nh++; else if (pid == 22) ng++
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
