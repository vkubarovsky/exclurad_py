// CLAS12 MC -> flat ntuple AS AN ANALYSIS SEES IT.  No truth matching anywhere
// in the selection.
//
//   run-groovy rec_to_ntuple.groovy <file.hipo> <max events> <out.f32> [skip] [hadron pid]
//
// WHY THIS EXISTS, BESIDE hipo_to_ntuple.groovy.  That one starts from the
// GENERATED particle and asks MC::GenMatch which reconstructed row belongs to
// it.  That is the right tool for an efficiency -- it is how we know the
// electron is found in 13.3 % of events -- but it is not what Story, or anyone
// on data, can do.  An analysis has REC::Particle and nothing else: it must
// pick the electron, the baryon and the photon pair itself, out of whatever the
// detector produced, including the 50 nA background.
//
// Choosing cuts on truth-matched quantities would choose cuts that cannot be
// applied.  Three places where the difference is not cosmetic:
//
//   * THE PHOTON PAIR.  hipo_to_ntuple picks the right two photons by their
//     MC::Lund parent.  An analysis picks, say, the two hardest -- and that
//     rule is right in only 52.8 % of our radiative events, because the pi0
//     decay is asymmetric and the radiated photon is often harder than a decay
//     photon (measured: 0.927 radiated against 0.907 and 0.011).
//   * BACKGROUND.  Extra clusters exist for the analysis and not for the match.
//   * MISIDENTIFICATION.  A proton called a pi+ is "found with the wrong pid"
//     to a matcher and simply absent to an analysis.
//
// So everything in fields [13..] below comes from REC::Particle alone.  The
// generated block [1..12] is carried for reference and MUST NOT be used to
// select anything.  Fields [43],[44] are the one deliberate exception: they are
// diagnostics that say where the TRUE photons ended up in the analysis-ordered
// list, which is what lets us measure how often a pairing rule is right.
//
// Record per event (NFLD floats), little-endian float32, same shape as
// hipo_to_ntuple so the same numpy loader works:
//   [0]       weight (MC::Event)
//   [1..12]   GENERATED px,py,pz for e, p, g1, g2   -- reference only
//   [13]      n_rec rows      [14] n(pid 11)   [15] n(pid = hadron)  [16] n(pid 22)
//   [17..21]  best electron : px py pz status chi2pid      (highest |p|, pid 11)
//   [22..27]  best baryon   : px py pz status chi2pid beta (highest |p|, pid = hadron)
//   [28..42]  five hardest photons, px py pz each, zero-padded
//   [43],[44] index in that photon list of the true g1 / g2, or -1  -- DIAGNOSTIC
//   [45]      electron vz    [46] RUN::config torus   [47] spare
//   [48]      1 if no electron candidate at all       [49] spare
import org.jlab.jnp.hipo4.data.*
import org.jlab.jnp.hipo4.io.*
import java.nio.*

def fname = args[0]
int maxev = args[1] as int
def oname = args[2]
int skip = args.size() > 3 ? args[3] as int : 0
int hpid = args.size() > 4 ? args[4] as int : 2212
final int NFLD = 50, NGAM = 5

def reader = new HipoReader(); reader.open(fname)
def ev = new Event()
def lund = reader.getBank("MC::Lund"), mcp = reader.getBank("MC::Particle"), mce = reader.getBank("MC::Event")
def gm = reader.getBank("MC::GenMatch"), rec = reader.getBank("REC::Particle")
def cfg = reader.getBank("RUN::config")
def os = new BufferedOutputStream(new FileOutputStream(oname), 1<<20)
def bb = ByteBuffer.allocate(NFLD*4).order(ByteOrder.LITTLE_ENDIAN)
float[] r = new float[NFLD]

int n = 0, nw = 0, nskip = 0, nnoe = 0, nfull = 0
int[] gi = new int[4]
long t0 = System.currentTimeMillis()
try {
  while (reader.hasNext() && nskip < skip) { reader.nextEvent(ev); nskip++ }
  while (reader.hasNext() && n < maxev) {
    reader.nextEvent(ev); n++
    ev.read(lund); ev.read(mcp); ev.read(mce); ev.read(gm); ev.read(rec); ev.read(cfg)
    Arrays.fill(r, 0f)
    r[0] = mce.getRows() > 0 ? mce.getFloat("weight", 0) : 1f

    // ---- generated block, reference only -------------------------------
    // Same photon-by-parent logic as hipo_to_ntuple: the radiated photon has
    // MC::Lund parent 0, the decay pair has the pi0's 1-based row.
    def src = mcp.getRows() > 0 ? mcp : lund
    int ie = -1, ip = -1, ig1 = -1, ig2 = -1
    double[][] pi0g = new double[2][3]; int npi0g = 0
    if (lund.getRows() > 0) {
      int pi0row = -1
      for (int i = 0; i < lund.getRows(); i++) if (lund.getInt("pid", i) == 111) { pi0row = i + 1; break }
      if (pi0row > 0) for (int i = 0; i < lund.getRows() && npi0g < 2; i++)
        if (lund.getInt("pid", i) == 22 && lund.getInt("parent", i) == pi0row) {
          pi0g[npi0g][0]=lund.getFloat("px",i); pi0g[npi0g][1]=lund.getFloat("py",i)
          pi0g[npi0g][2]=lund.getFloat("pz",i); npi0g++
        }
    }
    for (int i = 0; i < src.getRows(); i++) {
      int pid = src.getInt("pid", i)
      if (pid == 11 && ie < 0) { ie = i; continue }
      if (pid == hpid && ip < 0) { ip = i; continue }
      if (pid == 22) {
        if (npi0g == 2) {                       // keep only the decay pair
          int w = -1
          for (int q = 0; q < 2; q++)
            if (Math.abs(src.getFloat("px",i)-pi0g[q][0]) < 1e-5 &&
                Math.abs(src.getFloat("py",i)-pi0g[q][1]) < 1e-5 &&
                Math.abs(src.getFloat("pz",i)-pi0g[q][2]) < 1e-5) { w = q; break }
          if (w < 0) continue
        }
        if (ig1 < 0) ig1 = i; else if (ig2 < 0) ig2 = i
      }
    }
    gi[0]=ie; gi[1]=ip; gi[2]=ig1; gi[3]=ig2
    for (int k = 0; k < 4; k++) if (gi[k] >= 0) {
      r[1+3*k]   = src.getFloat("px", gi[k])
      r[1+3*k+1] = src.getFloat("py", gi[k])
      r[1+3*k+2] = src.getFloat("pz", gi[k])
    }

    // ---- everything below: REC::Particle ONLY --------------------------
    int ne=0, np=0, ng=0
    int be = -1, bp = -1
    double bpe = -1, bpp = -1
    int[] gidx = new int[NGAM]; double[] gmom = new double[NGAM]
    for (int q = 0; q < NGAM; q++) { gidx[q] = -1; gmom[q] = -1 }
    for (int i = 0; i < rec.getRows(); i++) {
      int pid = rec.getInt("pid", i)
      double px = rec.getFloat("px", i), py = rec.getFloat("py", i), pz = rec.getFloat("pz", i)
      double p = Math.sqrt(px*px + py*py + pz*pz)
      if (pid == 11)   { ne++; if (p > bpe) { bpe = p; be = i } }
      else if (pid == hpid) { np++; if (p > bpp) { bpp = p; bp = i } }
      else if (pid == 22) {
        ng++
        // insertion sort into the NGAM hardest
        for (int q = 0; q < NGAM; q++) {
          if (p > gmom[q]) {
            for (int z = NGAM-1; z > q; z--) { gmom[z]=gmom[z-1]; gidx[z]=gidx[z-1] }
            gmom[q] = p; gidx[q] = i; break
          }
        }
      }
    }
    r[13]=rec.getRows(); r[14]=ne; r[15]=np; r[16]=ng
    if (be >= 0) {
      r[17]=rec.getFloat("px",be); r[18]=rec.getFloat("py",be); r[19]=rec.getFloat("pz",be)
      r[20]=rec.getShort("status",be); r[21]=rec.getFloat("chi2pid",be)
      r[45]=rec.getFloat("vz",be)
    } else { nnoe++; r[48]=1f }
    if (bp >= 0) {
      r[22]=rec.getFloat("px",bp); r[23]=rec.getFloat("py",bp); r[24]=rec.getFloat("pz",bp)
      r[25]=rec.getShort("status",bp); r[26]=rec.getFloat("chi2pid",bp); r[27]=rec.getFloat("beta",bp)
    }
    for (int q = 0; q < NGAM; q++) if (gidx[q] >= 0) {
      r[28+3*q]   = rec.getFloat("px", gidx[q])
      r[28+3*q+1] = rec.getFloat("py", gidx[q])
      r[28+3*q+2] = rec.getFloat("pz", gidx[q])
    }
    // ---- DIAGNOSTIC ONLY: where did the TRUE photons land in that list? --
    // This is the only place truth enters, and nothing above depends on it.
    // It is what lets us ask how often "the two hardest" is the right pair.
    r[43] = -1f; r[44] = -1f
    for (int j = 0; j < gm.getRows(); j++) {
      int mci = gm.getShort("mcindex", j), pi = gm.getShort("pindex", j)
      if (pi < 0) continue
      for (int k = 2; k < 4; k++) {
        if (gi[k] < 0 || mci != gi[k]) continue
        for (int q = 0; q < NGAM; q++) if (gidx[q] == pi) { r[43 + (k-2)] = q; break }
      }
    }
    if (be >= 0 && bp >= 0 && gidx[1] >= 0) nfull++
    r[46] = cfg.getRows() > 0 ? cfg.getFloat("torus", 0) : 0f
    // [47] was startTime.  Dropped: the accessor prints "has type Float while
    // requested UNDEFINED" once PER EVENT and does not throw, so a try/catch
    // does not silence it -- five million lines of log for a field nothing
    // here needs.  The start-time story (no trigger electron -> TOF PID
    // collapses) is understood and does not need the number in the ntuple.
    r[47] = 0f
    bb.clear(); for (int k = 0; k < NFLD; k++) bb.putFloat(r[k])
    os.write(bb.array()); nw++
    if (n % 500000 == 0) { double dt=(System.currentTimeMillis()-t0)/1000.0; System.err.println("  $n events, ${(int)(n/dt)} ev/s") }
  }
} catch (Throwable e) {
  System.err.println("FATAL at event $n : " + e); e.printStackTrace(); os.close(); System.exit(2)
}
os.close()
double dt=(System.currentTimeMillis()-t0)/1000.0
println("DONE skipped $nskip read $n written $nw noElectron $nnoe fullTopology $nfull  ${(int)(n/dt)} ev/s -> $oname")
