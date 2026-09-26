// aao_gen CLAS12 MC -> flat binary ntuple (little-endian float32, fixed record).
//   run-groovy aao_ntuple.groovy <file.hipo> <max events> <out.f32> [skip] [hadron pid]
// The detected baryon defaults to the proton, 2212.  Pass 2112 for a neutron
// target: on a deuteron the struck neutron AND the spectator proton are both in
// the record, so a hardcoded 2212 silently picks the SPECTATOR and the ntuple
// describes the wrong particle without failing.  Found by the fermi session,
// 2026-09-26; it affects no proton-target sample, and every neutron one.
// Record per event (NFLD floats):
//   [0]      weight (MC::Event)
//   [1..12]  gen px,py,pz for e, p, g1, g2 (MC::Lund order)
//   for k in e,p,g1,g2 (block of 8 each, starting at 13):
//     found(0/1) pid charge status quality px py pz   -- REC::Particle row given by MC::GenMatch
//                                                       (one row per MC particle; pindex=-1 -> not found)
//   [45] n_e(rec pid 11)  [46] n_baryon(rec pid = hadron pid)  [47] n_gamma(rec pid 22)
//   [48] n_rec rows   [49] flag: MC::Particle row count != MC::Lund row count (status-0 rows in Lund)
import org.jlab.jnp.hipo4.data.*
import org.jlab.jnp.hipo4.io.*
import java.nio.*

def fname = args[0]
int maxev = args[1] as int
def oname = args[2]
int skip = args.size() > 3 ? args[3] as int : 0
int hpid = args.size() > 4 ? args[4] as int : 2212   // 2112 for a neutron target
final int NFLD = 50

def reader = new HipoReader(); reader.open(fname)
def ev = new Event()
def lund = reader.getBank("MC::Lund"), mcp = reader.getBank("MC::Particle"), mce = reader.getBank("MC::Event")
def gm = reader.getBank("MC::GenMatch"), rec = reader.getBank("REC::Particle")
def os = new BufferedOutputStream(new FileOutputStream(oname), 1<<20)
def bb = ByteBuffer.allocate(NFLD*4).order(ByteOrder.LITTLE_ENDIAN)
float[] r = new float[NFLD]

int n = 0, nw = 0, nbad = 0, nskip = 0, nmis = 0
int[] gi = new int[4]
long t0 = System.currentTimeMillis()
try {
  while (reader.hasNext() && nskip < skip) { reader.nextEvent(ev); nskip++ }
  while (reader.hasNext() && n < maxev) {
    reader.nextEvent(ev); n++
    ev.read(lund); ev.read(mcp); ev.read(mce); ev.read(gm); ev.read(rec)
    Arrays.fill(r, 0f)
    r[0] = mce.getRows() > 0 ? mce.getFloat("weight", 0) : 1f
    // generator: e, p, g1, g2.  MC::GenMatch.mcindex refers to MC::Particle rows, which hold only the
    // propagated (status 1) particles -- a LUND with a status-0 pi0 row (exclurad_py) has 5 Lund rows
    // but 4 MC::Particle rows.  So indices AND momenta are taken from MC::Particle; Lund is the fallback.
    def src = mcp.getRows() > 0 ? mcp : lund
    int ie=-1, ip=-1, ig1=-1, ig2=-1
    for (int i = 0; i < src.getRows(); i++) {
      int pid = src.getInt("pid", i)
      if (pid == 11 && ie < 0) ie = i
      else if (pid == hpid && ip < 0) ip = i
      else if (pid == 22) { if (ig1 < 0) ig1 = i; else if (ig2 < 0) ig2 = i }
    }
    if (ie < 0 || ip < 0 || ig1 < 0 || ig2 < 0) { nbad++; continue }
    gi[0]=ie; gi[1]=ip; gi[2]=ig1; gi[3]=ig2
    for (int k = 0; k < 4; k++) {
      r[1+3*k]   = src.getFloat("px", gi[k])
      r[1+3*k+1] = src.getFloat("py", gi[k])
      r[1+3*k+2] = src.getFloat("pz", gi[k])
    }
    // consistency flag: MC::Particle row count differs from Lund (status-0 rows present)
    if (mcp.getRows() > 0 && mcp.getRows() != lund.getRows()) r[49] = 1f
    if (r[49] > 0) nmis++
    // truth matching: MC::GenMatch row j refers to MC::Particle j (== Lund row j here)
    for (int j = 0; j < gm.getRows(); j++) {
      int mci = gm.getShort("mcindex", j), pi = gm.getShort("pindex", j)
      if (pi < 0 || pi >= rec.getRows()) continue
      for (int k = 0; k < 4; k++) {
        if (mci != gi[k]) continue
        int b = 13 + 8*k
        r[b] = 1f; r[b+1] = rec.getInt("pid", pi); r[b+2] = rec.getByte("charge", pi); r[b+3] = rec.getShort("status", pi)
        r[b+4] = gm.getFloat("quality", j)
        r[b+5] = rec.getFloat("px", pi); r[b+6] = rec.getFloat("py", pi); r[b+7] = rec.getFloat("pz", pi)
      }
    }
    int ne=0, np=0, ng=0
    for (int i = 0; i < rec.getRows(); i++) { int pid = rec.getInt("pid", i); if (pid==11) ne++; else if (pid==hpid) np++; else if (pid==22) ng++ }
    r[45]=ne; r[46]=np; r[47]=ng; r[48]=rec.getRows()
    bb.clear(); for (int k = 0; k < NFLD; k++) bb.putFloat(r[k])
    os.write(bb.array()); nw++
    if (n % 500000 == 0) { double dt=(System.currentTimeMillis()-t0)/1000.0; System.err.println("  $n events, ${(int)(n/dt)} ev/s, written $nw") }
  }
} catch (Throwable e) {
  System.err.println("FATAL at event $n : " + e); e.printStackTrace(); os.close(); System.exit(2)
}
os.close()
double dt=(System.currentTimeMillis()-t0)/1000.0
println("DONE skipped $nskip read $n written $nw badGen $nbad lundRowsDiffer $nmis  ${(int)(n/dt)} ev/s -> $oname")
