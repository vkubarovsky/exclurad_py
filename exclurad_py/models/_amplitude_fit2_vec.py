"""Векторная реализация _amplitude_fit2.structure -- та же математика на numpy.

Зачем она существует.  Квадратура радиационной поправки передаёт модели
МАССИВЫ узлов, а скалярная обёртка разворачивала их в питоновский цикл:
профиль 06.09 -- 343 тысячи вызовов structure() на ОДНО радиационное
событие, 217 секунд из 268.  Здесь тот же расчёт идёт по массиву целиком.

Правила, на которых это держится:
  * математика перенесена из _amplitude_fit2.py строка в строку; никакой
    новой физики, никаких упрощений;
  * эквивалентность скалярному коду проверяется тестом self_check() по
    случайным точкам во всех трёх каналах И на пороге -- расхождение
    масок или значений сверх допуска -- отказ, а не предупреждение;
  * ветка вычисленных фаз (p[35] <= -9, phase_table.npz) НЕ перенесена
    намеренно: у боевого вектора она мертва, а тащить сюда интерполяцию
    таблицы без работающего эталона значило бы переносить непроверяемое.
    Вектор с такой веткой получает отказ сразу, не тихую ошибку.

Скалярный модуль остаётся источником истины: он побайтово совпадает с
amplitudes.py из pi0eta_ampmodel_fit, которым фитились параметры.

03.10.2026: перенесён блок <Htilde> (слоты 37-42).  Включается только при
len(p) > 41, поэтому все 27- и 37-слотовые модели идут прежним путём.
"""
import math

import numpy as np

from . import _amplitude_fit2 as _S

Mp = _S.Mp
K_T, K_L = _S.K_T, _S.K_L
S2, S6 = _S.S2, _S.S6
_MM = {"pi0p": _S.Mpi0, "etap": _S.Meta, "pi0n": _S.Mpi0, "etan": _S.Meta}


def structure_arr(p, ch, t, xB, Q2):
    """Девять структурных функций по массивам точек.

    Возвращает (out, ok): out -- dict массивов, ok -- маска годных точек
    (ниже порога или tp >= 0 -- False, значения там нули).  Каналы и
    формулы -- ровно как в скалярном structure().
    """
    p = np.asarray(p, dtype=float)
    if len(p) > 36 and p[35] <= -9.0:
        raise NotImplementedError(
            "ветка вычисленных фаз (p[35] <= -9) не векторизована; "
            "считайте скалярным _amplitude_fit2")
    t = np.asarray(t, dtype=float)
    x = np.asarray(xB, dtype=float)
    q = np.asarray(Q2, dtype=float)
    t, x, q = np.broadcast_arrays(t, x, q)
    mM = _MM[ch]

    with np.errstate(all="ignore"):
        # --- tmin, как в скалярном tmin(): порог -- через маску ----------
        W2 = q * (1.0 / x - 1.0) + Mp * Mp
        W = np.sqrt(np.maximum(W2, 1e-30))
        E1 = (W2 + q + Mp * Mp) / (2.0 * W)
        P1 = np.sqrt(np.maximum(E1 * E1 - Mp * Mp, 0.0))
        E3 = (W2 - mM * mM + Mp * Mp) / (2.0 * W)
        e3sq = E3 * E3 - Mp * Mp
        P3 = np.sqrt(np.maximum(e3sq, 0.0))
        tmin = (q + mM * mM) ** 2 / (4.0 * W2) - (P1 - P3) ** 2
        tp = t - tmin
        # t_max, the backward edge.  Until 2026-10-03 only t < t_min was tested,
        # so the model was evaluated at |t| far beyond where the process exists
        # -- the radiative integral reached |t| = 14 at W2 = 1.23, against a
        # physical limit of 0.42.
        tmax = (q + mM * mM) ** 2 / (4.0 * W2) - (P1 + P3) ** 2
        ok = ((x > 0.0) & (x < 1.0) & (q > 0.0) & (e3sq >= 0.0)
              & (tp < 0.0) & (t >= tmax))

        # --- _flavour, строка в строку -----------------------------------
        L = np.log(np.where(x > 0.0, x, 1.0))
        HTu = p[0] * np.exp((p[1] + p[2] * L) * t) * q ** (p[3] / 2.0)
        HTd = p[4] * np.exp((p[5] + p[6] * L) * t) * q ** (p[7] / 2.0)
        if len(p) > 27:
            ETu = p[8] * np.exp((p[9] + p[10] * L) * t) * q ** (p[11] / 2.0)
            ETd = p[13] * np.exp((p[14] + p[12] * L) * t) * q ** (p[27] / 2.0)
            if len(p) > 31:
                ETu = ETu * x ** p[28] * (1.0 - x) ** p[29]
                ETd = ETd * x ** p[30] * (1.0 - x) ** p[31]
        else:
            ETu = p[8] * np.exp((p[9] + p[10] * L) * t + p[12] * t * t) \
                * q ** (p[11] / 2.0)
            ETd = p[13] * ETu * np.exp(p[14] * t)
        bxL = p[21] if len(p) > 21 else 0.0
        Lu = p[15] * np.exp((p[16] + bxL * L) * t) * q ** (p[17] / 2.0)
        Ld = p[18] * Lu
        # --- _htilde, строка в строку (2026-10-03) -------------------------
        has_ht = len(p) > 41
        if has_ht:
            Htu = p[37] * np.exp((p[38] + p[39] * L) * t) * q ** (p[40] / 2.0)
            Htd = p[41] * Htu

        # --- фазы u-d (живая ветка боевого вектора) -----------------------
        HTd_c = HTd.astype(complex)
        ETd_c = ETd.astype(complex)
        Ld_c = Ld.astype(complex)
        if len(p) > 36:
            HTd_c = HTd * np.exp(1j * p[34])
            ETd_c = ETd * np.exp(1j * p[35])
            Ld_c = Ld * np.exp(1j * p[36])
        # Htilde shares the Etilde u-d phase, p[36], as in the scalar module.
        if has_ht:
            Htd_c = Htd.astype(complex)
            if len(p) > 36:
                Htd_c = Htd * np.exp(1j * p[36])

        # --- _combine ------------------------------------------------------
        def comb(u, d, k):
            # Every channel is named explicitly and an unknown one RAISES.
            # The bare fall-through that used to end this function returned
            # the pi0n combination for anything unrecognised, so adding a
            # channel here would have silently produced pi0n numbers under
            # another channel's name instead of failing.
            if ch == "pi0p":
                return (2.0 * u + d) / (3.0 * S2)
            if ch == "etap":
                return (2.0 * u - d) / (3.0 * S6 * k)
            if ch == "pi0n":
                return (u + 2.0 * d) / (3.0 * S2)
            if ch == "etan":
                return (2.0 * d - u) / (3.0 * S6 * k)
            raise ValueError(ch)

        HT = comb(HTu, HTd_c, K_T)
        ET = comb(ETu, ETd_c, K_T)
        LL = comb(Lu, Ld_c, K_L)
        HTIL = comb(Htu, Htd_c, K_L) if has_ht else None

        # --- amplitudes() ---------------------------------------------------
        lam = W2 * W2 + q * q + Mp ** 4 + 2 * W2 * q - 2 * W2 * Mp * Mp \
            + 2 * q * Mp * Mp
        Lam = 16.0 * _S.PI * (W2 - Mp * Mp) * np.sqrt(np.maximum(lam, 0.0))
        A = 4.0 * _S.PI * _S.ALPHA / 2.0 / np.where(Lam > 0, Lam, 1.0) \
            / (q * q) * _S.HC2
        xi = x / (2.0 - x) * (1.0 + Mp * Mp / q)
        kin = -tp / (8.0 * Mp * Mp)

        D = np.sqrt(np.maximum(2.0 * A * (1.0 - xi * xi), 0.0)) * HT
        U01p = np.sqrt(np.maximum(A * kin, 0.0)) * ET
        Lmag = np.sqrt(np.maximum(A * kin, 0.0)) * LL
        delta = p[19] + p[20] * t
        T00m = Lmag * np.exp(1j * delta)
        if len(p) > 22:
            T01p = p[22] * np.abs(U01p) * np.exp(1j * p[23])
            U01c = U01p * np.exp(1j * p[24])
            T00p = p[25] * np.abs(Lmag) * np.exp(1j * p[26])
        else:
            T01p = np.zeros_like(T00m)
            U01c = U01p.astype(complex)
            T00p = np.zeros_like(T00m)
        if has_ht:
            # Longitudinal sector with the correct forward behaviour; the old
            # L block is re-read as <Etilde> and <Htilde> populates the
            # non-flip amplitude, which survives t' -> 0.
            xi2 = np.minimum(xi * xi, 0.999999)
            Ec = LL * np.exp(1j * delta)
            Hc = HTIL * np.exp(1j * p[26])
            nx = p[42] if len(p) > 42 else 0.0
            T00m = np.sqrt(np.maximum(A * kin, 0.0)) * (xi ** nx) * Ec
            T00p = np.sqrt(np.maximum(A * (1.0 - xi2), 0.0)) \
                * (Hc - xi2 / (1.0 - xi2) * Ec)
        T01m = D / 2.0

        # --- structure(): грамова таблица -----------------------------------
        def s2f(a, b):
            return np.abs(a) ** 2 + np.abs(b) ** 2

        def ipf(a1, a2, b1, b2):
            return np.conjugate(a1) * b1 + np.conjugate(a2) * b2

        sL = s2f(T00p, T00m)
        sT01 = s2f(T01p, T01m)
        sU01 = s2f(U01c, T01m)          # U01m = T01m = D/2
        uv = -S2 * ipf(T00p, T00m, T01p, T01m)
        uw = -S2 * ipf(T00p, T00m, U01c, T01m)
        vw = 2.0 * ipf(T01p, T01m, U01c, T01m)

    z = np.where(ok, 1.0, 0.0)
    out = dict(L=sL * z, T=(sT01 + sU01) * z, TT=(sT01 - sU01) * z,
               LT=uv.real * z, LTp=uv.imag * z,
               LTpLL=uw.real * z, LTUL=uw.imag * z,
               Tp=vw.real * z, TTUL=vw.imag * z)
    return out, ok


def self_check(p, n=4000, seed=7, tol=1e-10):
    """Эквивалентность скалярному модулю; возвращает худшее расхождение.

    Точки берутся и в коробке, и НАРОЧНО у порога и за ним: маски должны
    совпадать так же точно, как значения.  Скалярный код ниже порога либо
    возвращает None, либо поднимает ValueError (изъян rc_iter1, задокументи-
    рован при установке amp2026c) -- и то и другое должно означать ok=False.
    """
    rng = np.random.default_rng(seed)
    worst = 0.0
    for ch in ("pi0p", "etap", "pi0n", "etan"):
        t = -rng.uniform(0.001, 2.5, n)
        x = rng.uniform(0.05, 0.95, n)
        q = rng.uniform(0.8, 9.5, n)
        out, ok = structure_arr(p, ch, t, x, q)
        for i in range(n):
            try:
                s = _S.structure(p, ch, float(t[i]), float(x[i]), float(q[i]))
            except ValueError:
                s = None
            if s is None:
                if ok[i]:
                    raise AssertionError(
                        "маски разошлись в %s: скаляр None, вектор ok "
                        "при t=%.4f x=%.4f Q2=%.4f" % (ch, t[i], x[i], q[i]))
                continue
            if not ok[i]:
                raise AssertionError(
                    "маски разошлись в %s: скаляр считает, вектор нет "
                    "при t=%.4f x=%.4f Q2=%.4f" % (ch, t[i], x[i], q[i]))
            for k in out:
                a, b = float(out[k][i]), float(s[k])
                d = abs(a - b) / max(abs(b), 1e-25)
                if d > worst:
                    worst = d
                if d > tol:
                    raise AssertionError(
                        "%s[%s] расходится: %.15e против %.15e (отн. %.2e) "
                        "при t=%.4f x=%.4f Q2=%.4f"
                        % (ch, k, a, b, d, t[i], x[i], q[i]))
    return worst
