"""Розыгрыш массы векторного мезона по Брейту-Вигнеру.

Перенос `sample_bw` из diffrad_vm.f90 (строки 1422-1505), один в один,
включая нормировку огибающей и предел числа попыток.  Своего здесь ничего
нет намеренно: эталон существует, и расхождение с ним было бы ошибкой
переноса, а не улучшением.

Зачем это обязательно.  У rho ширина 150 МэВ при массе 775 -- девятнадцать
процентов.  Фиксированная масса кладёт все пи+пи- пары в одну точку там,
где в эксперименте широкий резонанс.  И дело не только в форме пика: масса
входит в границы по t, в порог по W и через них в область, где считается
радиационная поправка.  У phi ширина 4.27 МэВ -- на грани разрешения, но
механизм всё равно общий.  У J/psi 93 кэВ, и дельта-функция там точна.

Как устроен розыгрыш.  Замена M^2 = M0^2 + M0*Gamma*tan(theta) с
равномерным theta даёт ТОЧНЫЙ Брейт-Вигнер с постоянной шириной -- никакой
отбраковки не нужно.  Для rho поверх этого идёт отбраковка на бегущую
ширину p-волны:

    Gamma(M) = Gamma0 * (p(M)/p(M0))^3 * (M0/M)

Тонкость, которую фортран отмечает отдельным комментарием: постоянная
ширина НЕ всегда даёт огибающую выше бегущей вблизи M0, поэтому отношение
нормируется на свой максимум, найденный сканом.  Без этого отбраковка была
бы неверна там, где отношение больше единицы.
"""
import numpy as np

# Полные ширины, PDG (diffrad_vm.f90:1516 -- gamhad)
WIDTH = {"rho": 0.1502, "omega": 0.00849, "phi": 0.004266, "jpsi": 0.0000929}

# Масса продукта распада: она задаёт нижнюю границу 2*m и бегущую ширину
M_DAUGHTER = {"rho": 0.13957, "omega": 0.13957, "phi": 0.493677,
              "jpsi": 0.000511}

# Ниже этой ширины мезон считается дельта-функцией.  Порог из фортрана:
# он же автоматически отсекает J/psi с его 93 кэВ.
DELTA_WIDTH = 1.0e-3

RUNNING_WIDTH = ("rho", "omega")   # p-волна: Gamma ~ (p/p0)^3


def width(channel):
    """Полная ширина канала; 0 для неизвестного -- значит дельта-функция."""
    return WIDTH.get(channel, 0.0)


def _running_width(m, m0, gam0, p0, m_dau):
    p = np.sqrt(max(0.0, 0.25 * m * m - m_dau * m_dau))
    if p0 <= 0.0:
        return gam0
    return gam0 * (p / p0) ** 3 * (m0 / m)


def sample_mass(channel, m0, m_max_kin, rng, gam=None):
    """Масса мезона для одного события.

    m_max_kin -- кинематический потолок, sqrt(W2) - m_proton.
    Возвращает m0, если ширина ничтожна или места нет: вызывающий тогда
    сам решает, отбрасывать событие или нет.
    """
    gam = WIDTH.get(channel, 0.0) if gam is None else gam
    m_dau = M_DAUGHTER.get(channel, 0.13957)
    m_min = 2.0 * m_dau
    m_max = min(m_max_kin, m0 + 5.0 * gam)
    if m_max <= m_min or gam < DELTA_WIDTH:
        return m0

    th_min = np.arctan((m_min * m_min - m0 * m0) / (m0 * gam))
    th_max = np.arctan((m_max * m_max - m0 * m0) / (m0 * gam))
    p0 = np.sqrt(max(0.0, 0.25 * m0 * m0 - m_dau * m_dau))

    running = channel in RUNNING_WIDTH and p0 > 0.0
    ratio_max = 1.0
    if running:
        # Скан огибающей: постоянная ширина не всегда выше бегущей.
        th = np.linspace(th_min, th_max, 501)
        m2 = m0 * m0 + m0 * gam * np.tan(th)
        m = np.sqrt(np.maximum(m_min * m_min, m2))
        p = np.sqrt(np.maximum(0.0, 0.25 * m * m - m_dau * m_dau))
        gr = gam * (p / p0) ** 3 * (m0 / m)
        bw_run = 1.0 / ((m2 - m0 * m0) ** 2 + m0 * m0 * gr * gr)
        bw_con = 1.0 / ((m2 - m0 * m0) ** 2 + m0 * m0 * gam * gam)
        r = bw_run / bw_con * (gr / gam) * (m / m0)
        ratio_max = max(1.0, float(np.max(r))) * 1.01

    for _ in range(10000):
        theta = th_min + rng.random() * (th_max - th_min)
        m2 = m0 * m0 + m0 * gam * np.tan(theta)
        m = np.sqrt(max(m_min * m_min, m2))
        if not running:
            return float(m)
        gr = _running_width(m, m0, gam, p0, m_dau)
        bw_run = 1.0 / ((m2 - m0 * m0) ** 2 + m0 * m0 * gr * gr)
        bw_con = 1.0 / ((m2 - m0 * m0) ** 2 + m0 * m0 * gam * gam)
        ratio = bw_run / bw_con * (gr / gam) * (m / m0) / ratio_max
        if rng.random() <= ratio:
            return float(m)
    # Десять тысяч попыток без успеха: возвращаем номинал, как фортран.
    return m0
