// aao_gen CLAS12 MC -> flat binary ntuple (little-endian float32, fixed record).
//   run-groovy aao_ntuple.groovy <file.hipo> <max events> <out.f32> [skip] [hadron pid]
// The detected baryon defaults to the proton, 2212.  Pass 2112 for a neutron
// target: on a deuteron the struck neutron AND the spectator proton are both in
// the record, so a hardcoded 2212 silently picks the SPECTATOR and the ntuple
// describes the wrong particle without failing.  Found by the fermi session,
// 2026-09-26; it affects no proton-target sample, and every neutron one.
// Record per event (NFLD = 55 floats):
//   [0]      weight (MC::Event)
//   [1..12]  gen px,py,pz for e, p, g1, g2 (MC::Lund order)
//   for k in e,p,g1,g2 (block of 8 each, starting at 13):
//     found(0/1) pid charge status quality px py pz   -- REC::Particle row given by MC::GenMatch
//                                                       (one row per MC particle; pindex=-1 -> not found)
//   [45] n_e(rec pid 11)  [46] n_baryon(rec pid = hadron pid)  [47] n_gamma(rec pid 22)
//   [48] n_rec rows   [49] flag: MC::Particle row count != MC::Lund row count (status-0 rows in Lund)
//   [50]      GENERATED vertex z (MC::Lund), cm -- one per event
//   [51..54]  RECONSTRUCTED vertex z per slot k = 0..3 (REC::Particle), cm;
//             0 where the slot was not matched
//
// Added 2026-10-22, for the reason given in vm_to_ntuple.groovy: the record
// carried momenta only, so nothing could plot the vertex, and a Born arm sat
// at z = 0 for a day unseen.
import org.jlab.jnp.hipo4.data.*
import org.jlab.jnp.hipo4.io.*
import java.nio.*

def fname = args[0]
int maxev = args[1] as int
def oname = args[2]
int skip = args.size() > 3 ? args[3] as int : 0
int hpid = args.size() > 4 ? args[4] as int : 2212   // 2112 for a neutron target
final int NFLD = 55

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

    // WHICH TWO PHOTONS ARE THE PI0's.  Taking the first two in MC::Particle
    // order is wrong for a radiative sample: the LUND carries THREE photons --
    // the radiated one, written before the decay pair -- so the first two are
    // the radiated photon and one decay photon, and the other decay photon is
    // dropped.  Measured on the 2026-09-25 rad million: 56.7 % of events have
    // three photons, and the pair was wrong in 26.8 % of the sample, which put
    // the "g1" efficiency at 39 % against 70 % for born.
    //
    // Taking the two HARDEST is no better -- 52.8 % correct -- because the pi0
    // decay is asymmetric: one photon can be 11 MeV while the radiated one is
    // 0.93 GeV.  The truth is in MC::Lund's parent column, which survives GEMC:
    // decay photons carry the pi0's 1-based row index, the radiated one carries
    // 0.  Match those back to MC::Particle by momentum, since MC::Particle drops
    // the status-0 pi0 row and the indices do not line up.
    double[][] pi0gam = new double[2][3]
    int npi0gam = 0
    // GENERATED VERTEX, one per event.  Every row of MC::Lund carries the same
    // vz for an exclusive event, so the first row is as good as any.  GEMC
    // keeps it exactly -- BEAM_V/SPREAD_V are not applied to LUND input -- so
    // this is also the true vertex in the detector, and [51+k] minus this is
    // the vertex resolution of slot k.
    if (lund.getRows() > 0) r[50] = lund.getFloat("vz", 0)
    if (lund.getRows() > 0) {
      int pi0row = -1
      for (int i = 0; i < lund.getRows(); i++) if (lund.getInt("pid", i) == 111) { pi0row = i + 1; break }
      if (pi0row > 0) {
        for (int i = 0; i < lund.getRows() && npi0gam < 2; i++) {
          if (lund.getInt("pid", i) == 22 && lund.getInt("parent", i) == pi0row) {
            pi0gam[npi0gam][0] = lund.getFloat("px", i)
            pi0gam[npi0gam][1] = lund.getFloat("py", i)
            pi0gam[npi0gam][2] = lund.getFloat("pz", i)
            npi0gam++
          }
        }
      }
    }

    int ie=-1, ip=-1, ig1=-1, ig2=-1
    for (int i = 0; i < src.getRows(); i++) {
      int pid = src.getInt("pid", i)
      if (pid == 11 && ie < 0) ie = i
      else if (pid == hpid && ip < 0) ip = i
      else if (pid == 22) {
        int which = -1
        if (npi0gam == 2) {
          // keep only the photons the pi0 actually produced
          double px = src.getFloat("px", i), py = src.getFloat("py", i), pz = src.getFloat("pz", i)
          for (int j = 0; j < 2; j++) {
            if (Math.abs(px - pi0gam[j][0]) < 1e-5 && Math.abs(py - pi0gam[j][1]) < 1e-5
                && Math.abs(pz - pi0gam[j][2]) < 1e-5) { which = j; break }
          }
          if (which < 0) continue                 // the radiated photon: skip it
        }
        if (ig1 < 0) ig1 = i; else if (ig2 < 0) ig2 = i
      }
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
        r[51 + k] = rec.getFloat("vz", pi)
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
