"""
generador_recibos.py
====================

Genera imágenes sintéticas para el obligatorio de Deep Learning:

* POSITIVOS: recibos de sueldo, en 9 "familias" de diseño inspiradas en recibos
  uruguayos reales (estructura, no marca: todas las empresas, personas, RUT,
  C.I. y montos son inventados, y no se usan logos de instituciones reales).
* NEGATIVOS: documentos que NO son recibos de sueldo, en 11 categorías: factura
  (e-factura y factura de servicio tipo e-Ticket), estado_cuenta (cuenta bancaria y
  tarjeta de crédito), ticket, presupuesto, constancia_laboral, factura_servicios,
  planilla_horas, recibo_pago, carta, tributo_municipal y recibo_vacio (recibos de
  sueldo SIN completar: parecen recibos pero no tienen validez). Los documentos de la
  carpeta Doc_varios_no_recibos se usaron como ejemplo para la categoría que les
  corresponde. Comparten rasgos visuales con los recibos (tablas, montos, datos
  personales), para que el modelo no aprenda atajos tipo "tabla con plata = recibo".

IMPORTANTE: el modelo se entrena SOLO con estas imágenes sintéticas. Los documentos
reales (carpetas Recibos/Reales y Doc_varios_no_recibos) quedan para validar y testear.

Positivos y negativos pasan por EXACTAMENTE el mismo motor de dibujo, las mismas
fuentes y las mismas augmentations (foto de celular / escaneo / digital), así la
única diferencia sistemática entre clases es el contenido del documento.

Uso (en Colab o local):

    # dataset completo: 20 por familia de recibo (9 x 20 = 180) y 180 negativos
    python generador_recibos.py --salida datos_sinteticos --n-pos 20 --n-neg-total 180

    # muestra: 1 por familia/tipo, versión limpia + aumentada + hojas de contacto
    python generador_recibos.py --salida muestra --muestra

Salida:
    <salida>/positivos/<familia>_<i>.jpg
    <salida>/negativos/<tipo>_<i>.jpg
    <salida>/metadata.csv   (archivo, etiqueta, clase, familia, subtipo, modo_aug, semilla)

Dependencias: Pillow, numpy (y matplotlib solo para ubicar las fuentes DejaVu).
"""

import argparse
import csv
import io
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

# ---------------------------------------------------------------------------
# Fuentes: se usan Liberation (viene en Colab) y DejaVu (viene con matplotlib)
# ---------------------------------------------------------------------------
try:
    import matplotlib

    _MPL = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
except Exception:  # pragma: no cover
    _MPL = ""
_LIB = "/usr/share/fonts/truetype/liberation"

# (carpeta, regular, negrita, itálica)
_CANDIDATAS = {
    "sans": [
        (_LIB, "LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf", "LiberationSans-Italic.ttf"),
        (_LIB, "LiberationSansNarrow-Regular.ttf", "LiberationSansNarrow-Bold.ttf", "LiberationSansNarrow-Italic.ttf"),
        (_MPL, "DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf"),
    ],
    "serif": [
        (_LIB, "LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf", "LiberationSerif-Italic.ttf"),
        (_MPL, "DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf", "DejaVuSerif-Italic.ttf"),
    ],
    "mono": [
        (_LIB, "LiberationMono-Regular.ttf", "LiberationMono-Bold.ttf", "LiberationMono-Italic.ttf"),
        (_MPL, "DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf", "DejaVuSansMono-Oblique.ttf"),
    ],
}
FUENTES = {
    k: [c for c in v if os.path.exists(os.path.join(c[0], c[1]))] for k, v in _CANDIDATAS.items()
}
for k in FUENTES:
    if not FUENTES[k]:  # último recurso: cualquier sans disponible
        FUENTES[k] = FUENTES["sans"] or [(_MPL, "DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf")]
_CACHE_FUENTES = {}


def _font(path, size):
    key = (path, size)
    if key not in _CACHE_FUENTES:
        _CACHE_FUENTES[key] = ImageFont.truetype(path, size)
    return _CACHE_FUENTES[key]


# ---------------------------------------------------------------------------
# Datos ficticios
# ---------------------------------------------------------------------------
NOMBRES = ["Juan", "María", "José", "Ana", "Carlos", "Laura", "Diego", "Lucía", "Martín", "Valentina",
           "Sebastián", "Camila", "Pablo", "Florencia", "Nicolás", "Sofía", "Federico", "Mariana",
           "Gonzalo", "Carolina", "Rodrigo", "Natalia", "Andrés", "Paula", "Matías", "Gabriela",
           "Santiago", "Verónica", "Alejandro", "Cecilia", "Fernando", "Silvia", "Jorge", "Patricia",
           "Agustín", "Romina", "Leonardo", "Daniela", "Marcelo", "Inés", "Ignacio", "Lorena",
           "Washington", "Graciela", "Álvaro", "Beatriz", "Rubén", "Mónica", "Hugo", "Elena"]
APELLIDOS = ["Rodríguez", "González", "Fernández", "López", "Martínez", "Pérez", "García", "Sánchez",
             "Silva", "Pereira", "Suárez", "Ramírez", "Sosa", "Núñez", "Díaz", "Méndez", "Álvarez",
             "Acosta", "Cabrera", "Olivera", "Castro", "Romero", "Torres", "Benítez", "Viera",
             "Correa", "Da Silva", "De León", "Techera", "Larrosa", "Píriz", "Machado", "Bentancur",
             "Ferreira", "Clavijo", "Morales", "Duarte", "Medina", "Vázquez", "Rocha", "Cardozo",
             "Barreto", "Moreira", "Rivero", "Quintana", "Gutiérrez", "Bonilla", "Ifrán", "Arismendi",
             "Lemes", "Mautone", "Saravia", "Echeverría", "Invernizzi", "Pintos", "Cuadrado"]
CALLES = ["Av. 18 de Julio", "Av. Italia", "Rivera", "Bv. Artigas", "Colonia", "Mercedes", "Sarandí",
          "Rincón", "Zabala", "Av. Agraciada", "Garibaldi", "Av. Millán", "Av. 8 de Octubre",
          "Av. Brasil", "Ellauri", "Canelones", "Paysandú", "Uruguay", "San José", "Soriano",
          "Yaguarón", "Av. Rondeau", "Cerrito", "Piedras", "Misiones", "Juncal", "Constituyente",
          "Av. Gral. Flores", "Camino Carrasco", "Ruta 8 Km 17", "Ruta 1 Km 24", "Treinta y Tres",
          "Av. Larrañaga", "Av. Uruguay", "Lavalleja", "Florida", "Durazno", "Yí", "Cuareim"]
CIUDADES = ["Montevideo", "Montevideo", "Montevideo", "Canelones", "Maldonado", "Salto", "Paysandú",
            "Rivera", "Tacuarembó", "Florida", "Durazno", "Colonia del Sacramento", "Mercedes",
            "Minas", "Rocha", "Melo", "Artigas", "Treinta y Tres", "San José de Mayo", "Trinidad",
            "Fray Bentos", "Las Piedras", "Ciudad de la Costa", "Pando"]
EMP_RUBRO = ["Distribuidora", "Comercial", "Servicios", "Transportes", "Constructora", "Laboratorios",
             "Industrias", "Agropecuaria", "Consultora", "Inmobiliaria", "Frigorífico", "Metalúrgica",
             "Textil", "Logística", "Supermercados", "Panificadora", "Ferretería", "Clínica",
             "Estudio", "Cooperativa", "Imprenta", "Electromecánica", "Agencia", "Hotel", "Química"]
EMP_NUCLEO = ["del Este", "Oriental", "Rioplatense", "del Sur", "Atlántida", "Los Robles", "El Ceibo",
              "La Aurora", "del Plata", "Bella Vista", "Arroyo Seco", "Santa Lucía", "Punta Gorda",
              "del Litoral", "Los Pinos", "El Mirador", "Tres Cruces", "La Blanqueada", "Las Acacias",
              "Río Negro", "del Norte", "San Fernando", "La Estanzuela", "Costa Azul", "El Trébol"]
EMP_INVENT_SIL = ["tra", "vel", "mar", "lux", "nor", "tec", "sur", "pro", "vi", "lan", "ex", "co",
                  "dor", "fin", "ga", "ro", "len", "tis", "qua", "zon", "ber", "mi"]
EMP_SUFIJO = ["S.A.", "S.A.", "S.R.L.", "S.A.S.", "Ltda.", "& Cía."]
BANCOS = ["Banco del Litoral", "Banco Atlántico", "Banco Oriental de Crédito", "Banco Rioplatense",
          "Banco del Sur", "Banco Federal Uruguayo", "Caja de Ahorro del Plata", "Banco Comercial del Este"]
CARGOS = ["Auxiliar Administrativo", "Vendedor", "Cajero", "Peón", "Oficial Electricista", "Chofer",
          "Analista", "Técnico", "Recepcionista", "Operario", "Encargado", "Medio Oficial",
          "Ejecutivo de Cuentas", "Contador", "Docente", "Enfermero", "Cocinero", "Repartidor",
          "Programador", "Administrativo II", "Guardia", "Mozo", "Jefe de Sección", "Asistente"]
SECTORES = ["Administración", "Ventas", "Depósito", "Producción", "Logística", "Contaduría",
            "Informática", "Mantenimiento", "Atención al Cliente", "Compras", "RRHH", "Taller"]
MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Setiembre",
         "Octubre", "Noviembre", "Diciembre"]
ORGANISMOS = ["Junta Local de Villa Esperanza", "Administración Departamental de Puertos Altos",
              "Instituto Nacional de Estudios Rurales", "Servicio Descentralizado de Aguas del Norte",
              "Comisión Administradora de Parques", "Instituto Superior de Ciencias Agrarias",
              "Facultad de Estudios Ambientales", "Centro Regional de Formación Técnica",
              "Dirección de Obras de Cerro Bajo", "Agencia de Desarrollo del Litoral"]
BPC = 6864  # valor aproximado de la BPC (solo para que el IRPF dé montos plausibles)


def elegir(rng, seq):
    return seq[rng.randrange(len(seq))]


def ci_uy(rng):
    """Cédula uruguaya ficticia con dígito verificador válido."""
    n = rng.randint(1_000_000, 6_999_999)
    dig = [int(c) for c in f"{n:07d}"]
    dv = (10 - sum(d * p for d, p in zip(dig, [2, 9, 8, 7, 6, 3, 4])) % 10) % 10
    s = f"{n:07d}"
    estilo = rng.random()
    if estilo < 0.5:
        return f"{s[0]}.{s[1:4]}.{s[4:]}-{dv}"
    if estilo < 0.8:
        return f"{s}-{dv}"
    return f"{s}{dv}"


def nombre_empresa(rng):
    r = rng.random()
    if r < 0.45:
        base = f"{elegir(rng, EMP_RUBRO)} {elegir(rng, EMP_NUCLEO)}"
    elif r < 0.8:
        inv = "".join(elegir(rng, EMP_INVENT_SIL) for _ in range(rng.randint(2, 3))).capitalize()
        base = inv if rng.random() < 0.5 else f"{elegir(rng, EMP_RUBRO)} {inv}"
    else:
        base = f"{elegir(rng, APELLIDOS)} y {elegir(rng, APELLIDOS)}"
    return f"{base} {elegir(rng, EMP_SUFIJO)}"


def empresa(rng):
    nombre = nombre_empresa(rng)
    palabras = [p for p in nombre.replace(".", "").split() if p[0].isupper() and p not in ("S", "SA", "SRL", "SAS", "Ltda")]
    iniciales = "".join(p[0] for p in palabras[:3]) or "XX"
    return dict(
        nombre=nombre,
        iniciales=iniciales,
        direccion=f"{elegir(rng, CALLES)} {rng.randint(100, 4800)}",
        ciudad=elegir(rng, CIUDADES),
        rut=f"21{rng.randint(0, 9999999999):010d}",
        bps=str(rng.randint(100000, 9999999)),
        bse=str(rng.randint(10000, 999999)),
        mtss=str(rng.randint(10000, 99999999)),
        tel=f"{rng.choice(['2', '4'])}{rng.randint(100, 999)} {rng.randint(1000, 9999)}",
    )


def persona(rng):
    n1 = elegir(rng, NOMBRES)
    n2 = elegir(rng, NOMBRES) if rng.random() < 0.5 else ""
    a1, a2 = elegir(rng, APELLIDOS), elegir(rng, APELLIDOS)
    return dict(
        nombres=f"{n1} {n2}".strip(),
        apellidos=f"{a1} {a2}",
        ci=ci_uy(rng),
        nro=str(rng.randint(1, 99999)),
        cargo=elegir(rng, CARGOS),
        sector=elegir(rng, SECTORES),
        categoria=f"{rng.choice(['Cat.', 'Categoría', ''])} {rng.randint(1, 12)}".strip(),
        grupo=str(rng.randint(1, 24)),
        banco=elegir(rng, BANCOS),
        cuenta=f"{rng.randint(1000000, 999999999)}-{rng.randint(0, 99999):05d}",
        ingreso=(rng.randint(1, 28), rng.randint(1, 12), rng.randint(1995, 2024)),
    )


def fmt_num(x, estilo):
    if estilo == "uy":
        return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    if estilo == "uy0":
        return f"{round(x):,}".replace(",", ".")
    if estilo == "en":
        return f"{x:,.2f}"
    return f"{x:.2f}"


def fecha(d, m, a, sep="/"):
    return f"{d:02d}{sep}{m:02d}{sep}{a}"


# --- número a letras (para "Recibí conforme la suma de ...") ---
_UNI = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "once",
        "doce", "trece", "catorce", "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve",
        "veinte", "veintiuno", "veintidós", "veintitrés", "veinticuatro", "veinticinco", "veintiséis",
        "veintisiete", "veintiocho", "veintinueve"]
_DEC = ["", "", "", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta", "noventa"]
_CEN = ["", "ciento", "doscientos", "trescientos", "cuatrocientos", "quinientos", "seiscientos",
        "setecientos", "ochocientos", "novecientos"]


def _menor_mil(n):
    if n == 0:
        return ""
    if n == 100:
        return "cien"
    c, r = divmod(n, 100)
    if r < 30:
        t = _UNI[r] if r else ""
    else:
        d, u = divmod(r, 10)
        t = _DEC[d] + (" y " + _UNI[u] if u else "")
    return (_CEN[c] + " " + t).strip()


def en_letras(n):
    n = int(n)
    if n == 0:
        return "cero"
    mill, resto = divmod(n, 1_000_000)
    miles, uni = divmod(resto, 1000)
    partes = []
    if mill:
        partes.append("un millón" if mill == 1 else _menor_mil(mill) + " millones")
    if miles:
        partes.append("mil" if miles == 1 else _menor_mil(miles) + " mil")
    if uni:
        partes.append(_menor_mil(uni))
    s = " ".join(partes)
    return s.replace("veintiuno mil", "veintiún mil").replace("uno mil", "un mil")


# --- liquidación coherente ---
def irpf_mensual(nominal):
    tramos = [(7, 0), (10, .10), (15, .15), (30, .24), (50, .25), (75, .27), (115, .31), (1e9, .36)]
    base = nominal * 1.06 if nominal > 10 * BPC else nominal
    imp, prev = 0.0, 0.0
    for lim, t in tramos:
        hi = lim * BPC
        if base > prev:
            imp += (min(base, hi) - prev) * t
        prev = hi
    return imp


def liquidacion(rng, perfil=None, mes=None):
    perfil = perfil or rng.choice(["privado", "privado", "privado", "publico", "bancario", "jornalero"])
    mayus = rng.random() < 0.7
    sin_tildes = rng.random() < 0.3

    def L(s):
        if sin_tildes:
            s = s.translate(str.maketrans("ÁÉÍÓÚáéíóúÑñ", "AEIOUaeiouNn"))
        return s.upper() if mayus else s

    H, D = [], []  # haberes y descuentos: dict(concepto, detalle, cant, tasa, monto)
    if perfil == "jornalero":
        jornal = round(rng.uniform(1200, 3600), 2)
        dias = rng.randint(16, 26)
        H.append(dict(concepto=L("Jornales"), detalle=f"Días: {dias}", cant=dias, tasa=jornal, monto=round(dias * jornal, 2)))
        base = dias * jornal
    else:
        base = round(rng.uniform(24000, 170000), rng.choice([0, 2]))
        nombre = rng.choice(["Sueldo Básico", "Sueldo", "Sueldo Mensual", "Sueldo días trab.", "Salario Base"])
        H.append(dict(concepto=L(nombre), detalle="", cant=30, tasa=None, monto=base))
    vh = base / 200
    extras = [
        (0.45, "Prima por Antigüedad", lambda: (None, None, base * rng.uniform(.01, .08), f"Años: {rng.randint(1, 25)}")),
        (0.35, "Horas Extras", lambda: (h := rng.randint(2, 40), round(vh * 2, 2), h * vh * 2, "")),
        (0.12, "Horas Nocturnas", lambda: (h := rng.randint(4, 60), round(vh * .2, 2), h * vh * .2, "")),
        (0.25, "Presentismo", lambda: (None, None, base * rng.uniform(.04, .1), "")),
        (0.25, "Hogar Constituido", lambda: (None, None, rng.uniform(1200, 2900), "")),
        (0.12, "Prima por Hijo", lambda: (k := rng.randint(1, 3), None, k * rng.uniform(800, 1500), "")),
        (0.18, "Feriado Pago", lambda: (1, round(base / 15, 2), base / 15, "")),
        (0.25, "Comisiones", lambda: (None, None, base * rng.uniform(.03, .35), "")),
        (0.18, "Compensación Especial", lambda: (None, None, base * rng.uniform(.05, .4), "")),
        (0.10, "Salario Vacacional", lambda: (d := rng.randint(10, 25), None, base / 30 * d, f"Días: {d}")),
        (0.15, "Partida Alimentación", lambda: (None, None, rng.uniform(2000, 6000), "")),
        (0.12, "Reintegro Mutual", lambda: (None, None, rng.uniform(2500, 5500), "")),
        (0.10, "Incentivo por Productividad", lambda: (None, None, base * rng.uniform(.02, .15), "")),
    ]
    if mes in (6, 12) and rng.random() < 0.4:
        extras.append((1.0, "Aguinaldo", lambda: (None, None, base * rng.uniform(.4, .5), "")))
    for p, nom, f in extras:
        if rng.random() < p:
            cant, tasa, monto, det = f()
            H.append(dict(concepto=L(nom), detalle=det, cant=cant, tasa=tasa, monto=round(monto, 2)))
    nominal = sum(h["monto"] for h in H)

    jub = 0.175 if perfil == "bancario" and rng.random() < .7 else 0.15
    jub_nombre = rng.choice(["Aporte Jubilatorio", "Montepío", "Aporte Jub."]) if perfil != "privado" else rng.choice(["Aporte Jubilatorio", "Aporte Jub. BPS", "Montepío"])
    D.append(dict(concepto=L(f"{jub_nombre} {str(round(jub * 100, 1)).replace('.0', '')}%"), detalle="", cant=None, tasa=jub, monto=nominal * jub))
    fon = rng.choice([3, 4.5, 4.5, 5, 6, 8])
    if rng.random() < 0.35 and fon > 3:
        D.append(dict(concepto=L("FONASA Básico 3%"), detalle="", cant=None, tasa=.03, monto=nominal * .03))
        D.append(dict(concepto=L(f"FONASA Adicional {str(fon - 3).replace('.0', '')}%"), detalle="", cant=None, tasa=(fon - 3) / 100, monto=nominal * (fon - 3) / 100))
    else:
        D.append(dict(concepto=L(rng.choice(["FONASA", "Seguro Enfermedad", "SNIS"]) + f" {str(fon).replace('.0', '')}%"), detalle="", cant=None, tasa=fon / 100, monto=nominal * fon / 100))
    D.append(dict(concepto=L(rng.choice(["FRL", "Fondo Rec. Laboral", "F.R.L."]) + " 0,125%"), detalle="", cant=None, tasa=.00125, monto=nominal * .00125))
    irpf = max(0.0, irpf_mensual(nominal) - 0.1 * (nominal * (jub + fon / 100)))
    D.append(dict(concepto=L("IRPF") if rng.random() < .6 else L("I.R.P.F."), detalle="", cant=None, tasa=None, monto=irpf))
    opcionales = [
        (0.2, "Cuota Sindical", nominal * .01),
        (0.15, "Adelanto", nominal * rng.uniform(.05, .25)),
        (0.2, f"Préstamo {rng.choice(['Cooperativa', 'ANV', 'Banco', 'Financiera'])}", rng.uniform(1500, 9000)),
        (0.15, "Seguro de Vida", rng.uniform(200, 900)),
        (0.1, "Cuota Club Social", rng.uniform(300, 1200)),
        (0.08, "Retención Judicial", nominal * rng.uniform(.05, .2)),
    ]
    for p, nom, monto in opcionales:
        if rng.random() < p:
            D.append(dict(concepto=L(nom), detalle="", cant=None, tasa=None, monto=monto))
    for x in D:
        x["monto"] = round(x["monto"], 2)
    th = round(sum(h["monto"] for h in H), 2)
    td = round(sum(d["monto"] for d in D), 2)
    # códigos de concepto
    esquema = rng.choice(["seq", "sparse", "alfa"])
    cods = []
    for i in range(len(H) + len(D)):
        es_d = i >= len(H)
        if esquema == "seq":
            cods.append(str(i + 1))
        elif esquema == "sparse":
            cods.append(str((rng.randint(30, 99) * 100 + i) if es_d else (rng.randint(1, 60) * (i + 1))))
        else:
            cods.append(("DT" if es_d else "HA") + f"{rng.randint(1, 99):02d}")
    if esquema == "sparse":
        cods = sorted(cods[: len(H)], key=int) + sorted(cods[len(H):], key=int)
    for x, c in zip(H + D, cods):
        x["cod"] = c
    return dict(H=H, D=D, total_h=th, total_d=td, liquido=round(th - td, 2), perfil=perfil, irpf=irpf, nominal=nominal)


def datos(rng, perfil=None):
    anio = rng.randint(2019, 2026)
    mes = rng.randint(1, 11 if anio == 2026 else 12)
    emp, per = empresa(rng), persona(rng)
    if per["ingreso"][2] >= anio:
        per["ingreso"] = (per["ingreso"][0], per["ingreso"][1], anio - rng.randint(1, 10))
    liq = liquidacion(rng, perfil, mes)
    mp, ap = (mes % 12) + 1, anio + (1 if mes == 12 else 0)
    return dict(
        emp=emp, per=per, liq=liq, anio=anio, mes=mes, mes_nombre=MESES[mes - 1],
        fecha_pago=fecha(rng.randint(1, 10), mp, ap),
        tipo=rng.choice(["Mensual", "Mensual", "Mensual", "Quincenal", "Complementaria"]),
        fmt=rng.choice(["uy", "uy", "uy", "uy0", "plain"]),
    )


# ---------------------------------------------------------------------------
# Lienzo de dibujo
# ---------------------------------------------------------------------------
PAPELES = [(255, 255, 255)] * 6 + [(252, 251, 246), (248, 247, 240), (250, 250, 250), (246, 244, 236)]


class Lienzo:
    def __init__(self, rng, w, h, papel=None, tinta=None):
        self.rng = rng
        self.w, self.h = w, h
        self.img = Image.new("RGB", (w, h), papel or elegir(rng, PAPELES))
        self.d = ImageDraw.Draw(self.img)
        g = rng.randint(0, 45)
        self.tinta = tinta or (g, g, g + rng.randint(0, 15))
        self.fams = {k: elegir(rng, v) for k, v in FUENTES.items()}

    def fuente(self, size, bold=False, estilo="sans", italic=False):
        c = self.fams[estilo]
        archivo = c[3] if italic else (c[2] if bold else c[1])
        return _font(os.path.join(c[0], archivo), max(6, int(size)))

    def ancho(self, s, size, bold=False, estilo="sans"):
        return self.d.textlength(str(s), font=self.fuente(size, bold, estilo))

    def t(self, x, y, s, size=13, bold=False, align="l", estilo="sans", color=None, italic=False):
        s = str(s)
        f = self.fuente(size, bold, estilo, italic)
        w = self.d.textlength(s, font=f)
        if align == "r":
            x -= w
        elif align == "c":
            x -= w / 2
        self.d.text((x, y), s, font=f, fill=color or self.tinta)
        return w

    def l(self, x1, y1, x2, y2, w=1, color=None):
        self.d.line([(x1, y1), (x2, y2)], fill=color or self.tinta, width=w)

    def r(self, x1, y1, x2, y2, w=1, color=None, fill=None, radio=0):
        if radio:
            self.d.rounded_rectangle([x1, y1, x2, y2], radius=radio, outline=color or self.tinta, width=w, fill=fill)
        else:
            self.d.rectangle([x1, y1, x2, y2], outline=color or self.tinta, width=w, fill=fill)

    def punteada(self, x1, y, x2, paso=5, color=None):
        for x in range(int(x1), int(x2), paso * 2):
            self.l(x, y, min(x + paso, x2), y, 1, color)

    def parrafo(self, x, y, ancho, texto, size=12, interlinea=1.45, estilo="sans", color=None, italic=False, bold=False):
        palabras, linea = texto.split(), ""
        for p in palabras:
            prueba = (linea + " " + p).strip()
            if self.ancho(prueba, size, bold, estilo) > ancho and linea:
                self.t(x, y, linea, size, bold, estilo=estilo, color=color, italic=italic)
                y += size * interlinea
                linea = p
            else:
                linea = prueba
        if linea:
            self.t(x, y, linea, size, bold, estilo=estilo, color=color, italic=italic)
            y += size * interlinea
        return y

    def capa(self, capa_rgba):
        self.img = Image.alpha_composite(self.img.convert("RGBA"), capa_rgba).convert("RGB")
        self.d = ImageDraw.Draw(self.img)


def tabla(lz, x, y, w, cols, filas, alto_fila=18, size=11, bordes="full", header_fill=None,
          alto_min=None, estilo="sans", header_bold=True, color_borde=None, zebra=None, header=True):
    """Dibuja una tabla. cols = [(titulo, fraccion_ancho, align)]. Devuelve y final."""
    xs = [x]
    for _, frac, _ in cols:
        xs.append(xs[-1] + frac * w)
    cb = color_borde or lz.tinta
    hh = alto_fila + 4 if header else 0
    if header:
        if header_fill:
            lz.r(x, y, x + w, y + hh, 1, cb, fill=header_fill)
        for (tit, _, al), x0, x1 in zip(cols, xs, xs[1:]):
            tx = x0 + 4 if al == "l" else (x1 - 4 if al == "r" else (x0 + x1) / 2)
            lz.t(tx, y + 3, tit, size, header_bold, al, estilo)
    y_body = y + hh
    alto_body = max(alto_min or 0, len(filas) * alto_fila + 6)
    for i, fila in enumerate(filas):
        yy = y_body + 3 + i * alto_fila
        if zebra and i % 2 == 1:
            lz.d.rectangle([x + 1, yy - 1, x + w - 1, yy + alto_fila - 2], fill=zebra)
        for (tit, _, al), x0, x1, val in zip(cols, xs, xs[1:], fila):
            if val is None or val == "":
                continue
            tx = x0 + 4 if al == "l" else (x1 - 4 if al == "r" else (x0 + x1) / 2)
            lz.t(tx, yy, val, size, False, al, estilo)
    y_fin = y_body + alto_body
    if bordes in ("full", "vert"):
        lz.r(x, y, x + w, y_fin, 1, cb)
        for xx in xs[1:-1]:
            lz.l(xx, y, xx, y_fin, 1, cb)
    if bordes in ("full", "header", "vert") and header:
        lz.l(x, y + hh, x + w, y + hh, 1, cb)
    if bordes == "header" and header:
        lz.l(x, y, x + w, y, 1, cb)
    return y_fin


def logo(lz, x, y, s, iniciales, color):
    """Logo genérico inventado (formas simples + iniciales)."""
    rng = lz.rng
    tipo = rng.randrange(5)
    if tipo == 0:
        lz.d.ellipse([x, y, x + s, y + s], fill=color)
        lz.t(x + s / 2, y + s * .28, iniciales[:2], s * .38, True, "c", color=(255, 255, 255))
    elif tipo == 1:
        lz.d.rectangle([x, y, x + s, y + s], fill=color)
        lz.t(x + s / 2, y + s * .28, iniciales[:2], s * .38, True, "c", color=(255, 255, 255))
    elif tipo == 2:
        lz.d.polygon([(x, y + s), (x + s / 2, y), (x + s, y + s)], fill=color)
    elif tipo == 3:
        lz.d.ellipse([x, y, x + s * .7, y + s * .7], outline=color, width=4)
        lz.d.ellipse([x + s * .3, y + s * .3, x + s, y + s], outline=color, width=4)
    else:
        for i in range(4):
            lz.d.rectangle([x + i * s / 4, y + s - (i + 1) * s / 4, x + i * s / 4 + s / 5, y + s], fill=color)


def firma(lz, x, y, w, h, color=(30, 40, 120)):
    rng = lz.rng
    n = 60
    pts = []
    f1, f2 = rng.uniform(2, 5), rng.uniform(5, 11)
    a1, a2 = rng.uniform(.2, .5), rng.uniform(.1, .3)
    for i in range(n):
        t = i / (n - 1)
        px = x + w * t + w * .06 * math.sin(t * f2 * math.pi)
        py = y + h / 2 + h * a1 * math.sin(t * f1 * math.pi) + h * a2 * math.cos(t * f2 * math.pi)
        pts.append((px, py))
    lz.d.line(pts, fill=color, width=2, joint="curve")
    lz.d.line([(x + w * .1, y + h * .9), (x + w * .8, y + h * .75)], fill=color, width=2)


def sello(lz, cx, cy, r, textos, color=(70, 40, 140)):
    rng = lz.rng
    capa = Image.new("RGBA", lz.img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(capa)
    a = rng.randint(110, 190)
    col = color + (a,)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=3)
    d.ellipse([cx - r + 7, cy - r + 7, cx + r - 7, cy + r - 7], outline=col, width=1)
    f = lz.fuente(r * .22, True)
    for i, tx in enumerate(textos[:3]):
        w = d.textlength(tx, font=f)
        d.text((cx - w / 2, cy - r * .35 + i * r * .28), tx, font=f, fill=col)
    capa = capa.rotate(rng.uniform(-30, 30), center=(cx, cy))
    lz.capa(capa)


# ---------------------------------------------------------------------------
# FAMILIAS DE RECIBOS (positivos)
# ---------------------------------------------------------------------------
ACENTOS = [(25, 55, 125), (20, 90, 60), (80, 80, 80), (120, 30, 30), (0, 0, 0), (30, 90, 140), (90, 50, 120)]


def _filas_liq(D, con_detalle=True, con_noret=False, con_cant=False, con_tasa=False):
    n = lambda v: fmt_num(v, D["fmt"])
    filas = []
    for h in D["liq"]["H"]:
        f = [h["cod"], h["concepto"]]
        if con_detalle:
            f.append(h["detalle"])
        if con_cant:
            f.append("" if h["cant"] is None else fmt_num(h["cant"], "plain").replace(".00", ",00"))
        if con_tasa:
            f.append("" if h["tasa"] is None else fmt_num(h["tasa"], D["fmt"]))
        f += [n(h["monto"]), ""]
        if con_noret:
            f.append("")
        filas.append(f)
    for dd in D["liq"]["D"]:
        f = [dd["cod"], dd["concepto"]]
        if con_detalle:
            f.append(dd["detalle"])
        if con_cant:
            f.append("")
        if con_tasa:
            f.append("")
        f += ["", n(dd["monto"])]
        if con_noret:
            f.append("")
        filas.append(f)
    return filas


def fam_bloques(rng):
    """Estilo 'sistema web': cajas con borde, tabla Haberes/Descuentos/No retenidos, caja IRPF."""
    W, H = 900, 1273
    lz, D = Lienzo(rng, W, H), datos(rng, rng.choice([None, "bancario", "publico"]))
    n = lambda v: fmt_num(v, D["fmt"])
    acc = elegir(rng, ACENTOS)
    m = rng.randint(30, 50)
    y = m
    radio = rng.choice([0, 0, 6, 10])
    if rng.random() < .85:
        logo(lz, m, y, 44, D["emp"]["iniciales"], acc)
        lz.t(m + 54, y + 4, D["emp"]["nombre"].upper(), 15, True, color=acc)
        lz.t(m + 54, y + 24, rng.choice(["Recibo de sueldo", "Liquidación de haberes", "Recibo de haberes", ""]), 11, color=acc)
        y += 64
    E, P, L = D["emp"], D["per"], D["liq"]
    lz.r(m, y, W - m, y + 66, 1, acc, radio=radio)
    lz.t(m + 10, y + 8, f"Liquidación: {D['tipo']}", 12, True)
    lz.t(W - m - 10, y + 8, f"{D['mes_nombre']} {D['anio']}", 12, True, "r")
    lz.t(m + 10, y + 27, f"Descripción: {rng.randint(1, 9)} - {rng.choice(['Presupuesto', 'Haberes', 'Liquidación general', 'Planilla'])}", 11)
    lz.t(W * .55, y + 27, f"Nº de Cuenta: {P['cuenta']}", 11)
    lz.t(m + 10, y + 46, f"Fecha de pago: {D['fecha_pago']}", 11)
    lz.t(W * .55, y + 46, f"Lugar de pago: {P['banco']}", 11)
    y += 78
    mid = rng.randint(int(W * .52), int(W * .64))
    hb = 122
    lz.r(m, y, mid - 6, y + hb, 1, acc, radio=radio)
    lz.r(mid, y, W - m, y + hb, 1, acc, radio=radio)
    izq = [("Nº Funcionario/a", P["nro"]), ("Nombre", f"{P['apellidos']}, {P['nombres']}".upper()),
           ("C.I.", P["ci"]), ("Fecha de ingreso", fecha(*P["ingreso"])),
           ("Categoría", P["cargo"]), ("Grupo", P["grupo"])]
    for i, (k, v) in enumerate(izq):
        lz.t(m + 10, y + 8 + i * 18, f"{k}:", 11, True)
        lz.t(m + 10 + lz.ancho(k + ": ", 11, True), y + 8 + i * 18, v, 11)
    der = [("", E["nombre"]), ("", E["direccion"]), ("RUT", E["rut"]), ("B.P.S.", E["bps"]), ("B.S.E.", E["bse"]), ("MTSS", E["mtss"])]
    for i, (k, v) in enumerate(der):
        txt = f"{k}: {v}" if k else v
        lz.t(mid + 10, y + 8 + i * 18, txt, 11, i < 2)
    y += hb + 14
    noret = rng.random() < .6
    cols = [("Código", .09, "l"), ("Concepto", .34, "l"), ("Detalle", .17, "l"), ("Haberes", .14, "r"), ("Descuentos", .13, "r")]
    if noret:
        cols.append(("No retenidos", .13, "r"))
    tot = sum(c[1] for c in cols)
    cols = [(a, b / tot, c) for a, b, c in cols]
    filas = _filas_liq(D, con_noret=noret)
    hf = rng.choice([None, (235, 238, 245), (240, 240, 240)])
    y = tabla(lz, m, y, W - 2 * m, cols, filas, 19, 11, "full", hf, alto_min=rng.randint(300, 480), color_borde=acc)
    xs = [m]
    for c in cols:
        xs.append(xs[-1] + c[1] * (W - 2 * m))
    lz.t(xs[3] - 8, y + 8, "Totales", 12, True, "r")
    lz.t(xs[4] - 4, y + 8, n(L["total_h"]), 12, False, "r")
    lz.t(xs[5] - 4, y + 8, n(L["total_d"]), 12, False, "r")
    if noret:
        lz.t(xs[6] - 4, y + 8, n(0), 12, False, "r")
    y += 40
    lz.r(m + 120, y, m + 420, y + 28, 1, acc)
    lz.t(m + 130, y + 6, "Líquido a cobrar", 13, True)
    lz.t(m + 410, y + 6, n(L["liquido"]), 13, True, "r")
    y += 48
    if rng.random() < .75:
        c5 = [("Monto Renta", .2, "l"), ("Renta Inc", .2, "l"), ("IRPF Renta", .2, "l"), ("Monto Deducciones", .2, "l"), ("IRPF Deducciones", .2, "l")]
        dedu = L["nominal"] * .2
        fila = [n(L["nominal"]), n(L["nominal"] * 1.06), n(L["irpf"]), n(dedu), n(dedu * .08)]
        y = tabla(lz, m, y, W - 2 * m, c5, [fila], 20, 10, "full", None, color_borde=acc) + 16
    lz.parrafo(m, y, W - 2 * m, "La Institución declara haber efectuado los aportes de seguridad social correspondientes a los haberes liquidados al funcionario el mes anterior.", 10, italic=True)
    lz.t(W - m, H - m - 14, f"{E['nombre']} - Confidencial", 9, False, "r")
    return lz.img


def fam_sistema_viejo(rng):
    """Sistema viejo, tipografía monoespaciada, pocas líneas, 'Recibí conforme la suma de...'."""
    W, H = 900, rng.randint(900, 1273)
    lz, D = Lienzo(rng, W, H), datos(rng)
    E, P, L = D["emp"], D["per"], D["liq"]
    n = lambda v: fmt_num(v, D["fmt"])
    es = "mono"
    s = rng.choice([11, 12, 13])
    m = rng.randint(30, 60)
    y = m
    lz.t(m, y, f"Recibo No.   {rng.randint(1, 999)}    Fecha de Pago {D['fecha_pago']}", s, estilo=es)
    lz.t(W * .62, y, "Datos de la Empresa", s, estilo=es)
    y += s * 1.6
    lz.t(m, y, f"Funcionario  {P['nro']}   {P['apellidos'].split()[0].upper()} {P['nombres'].split()[0].upper()}", s, estilo=es)
    lz.t(W * .62, y, E["nombre"].upper()[:28], s, estilo=es)
    y += s * 1.6
    lz.t(m, y, f"Lugar de Pago {rng.randint(1, 300)}   {P['banco'].upper()[:22]}", s, estilo=es)
    lz.t(W * .62, y, E["direccion"], s, estilo=es)
    y += s * 1.8
    lz.t(m, y, " Ingreso       C.I.        Cargo", s, estilo=es)
    lz.t(W * .55, y, "R.U.C.          B.S.E.   B.P.S.", s, estilo=es)
    y += s * 1.3
    lz.t(m, y, f"{fecha(*P['ingreso'])}  {P['ci']}  {P['cargo'].upper()[:20]}", s, estilo=es)
    lz.t(W * .55, y, f"{E['rut']}   {E['bse'][:6]:<6}   {E['bps']}", s, estilo=es)
    y += s * 2.6
    con_v = rng.random() < .6
    cols = [("", .07, "l"), ("C o n c e p t o", .33, "l"), ("Unidades", .11, "r"), ("Tasa", .1, "r"), ("Haberes", .14, "r"), ("Descuentos", .13, "r"), ("No Retenido", .12, "r")]
    filas = []
    for h in L["H"]:
        filas.append([h["cod"], h["concepto"], "" if h["cant"] is None else f"{h['cant']:.2f}".replace(".", ","),
                      "" if h["tasa"] is None else f"{h['tasa']:.3f}".replace(".", ","), n(h["monto"]), "", ""])
    for d in L["D"]:
        filas.append([d["cod"], d["concepto"], "", "", "", n(d["monto"]), n(0)])
    y0 = y
    y = tabla(lz, m, y, W - 2 * m, cols, filas, s + 6, s, "vert" if con_v else "none", None, estilo=es, header_bold=False, alto_min=rng.randint(180, 320))
    y += 8
    lz.t(W * .48, y, "Totales", s, estilo=es)
    lz.t(W * .74, y, n(L["total_h"]), s, estilo=es, align="r")
    lz.t(W * .87, y, n(L["total_d"]), s, estilo=es, align="r")
    lz.t(W - m - 4, y, n(0), s, estilo=es, align="r")
    y += s * 2
    lz.t(W * .45, y, f"Líquido   *****{n(L['liquido'])}", s + 1, True, estilo=es)
    y += s * 1.8
    lz.t(m, y, "Recibí conforme la suma de PESOS URUGUAYOS", s, estilo=es)
    lz.parrafo(W * .55, y, W * .4, f"{en_letras(L['liquido'])} con {int(round((L['liquido'] % 1) * 100)):02d}/100", s, 1.3, es)
    y += s * 4
    lz.t(m + 30, y, "Monto Renta      Renta Inc.     IRPF Renta    Monto Deducciones   IRPF Deducciones", s - 1, estilo=es)
    y += s * 1.3
    lz.t(m + 30, y, f"{n(L['nominal']):>12}   {n(L['nominal'] * 1.06):>12}   {n(L['irpf']):>11}   {n(L['nominal'] * .2):>12}       {n(L['nominal'] * .016):>10}", s - 1, estilo=es)
    y += s * 2
    lz.t(m, y, f"Correspondiente a la Liquidación:   {D['tipo']}     {D['mes_nombre'].upper()} / {D['anio']}", s, estilo=es)
    y += s * 1.6
    lz.t(W * .5, y, "FIRMA: ______________________________", s, estilo=es)
    y += s * 2
    lz.t(m, y, "LA EMPRESA DECLARA ESTAR AL DÍA CON LOS APORTES A B.P.S.", s, estilo=es)
    lz.t(m, y + s * 1.4, "Comunicar inmediatamente cualquier discrepancia con los datos de este recibo.", s, estilo=es)
    return lz.img


def fam_generico_limpio(rng):
    """Recibo genérico limpio, apaisado, LÍQUIDO A COBRAR en caja."""
    W, H = 1000, rng.randint(760, 900)
    lz, D = Lienzo(rng, W, H), datos(rng, "privado")
    E, P, L = D["emp"], D["per"], D["liq"]
    n = lambda v: fmt_num(v, D["fmt"])
    m = rng.randint(20, 40)
    lz.r(4, 4, W - 5, H - 5, 1) if rng.random() < .5 else None
    y = m
    lz.t(m, y, E["nombre"].upper(), 26, True)
    lz.t(m, y + 38, f"Dirección: {E['direccion']}, {E['ciudad']}", 14)
    lz.t(m, y + 58, f"REMUNERACIÓN {P['sector'].upper()} {D['mes_nombre'].upper()} {D['anio']}", 12)
    for i, (k, v) in enumerate([("R.U.T.", E["rut"]), ("B.S.E.", E["bse"]), ("B.P.S.", E["bps"]), ("MTSS", E["mtss"])]):
        lz.t(W * .62, y + i * 20, f"{k}: {v}", 15)
    y += 92
    lz.l(0, y, W, y, 2)
    y += 16
    lz.t(m, y, "Nombre: ", 17, True)
    lz.t(m + lz.ancho("Nombre: ", 17, True), y, f"{P['nombres']} {P['apellidos']}", 17)
    y += 40
    izq = [("Sector", P["sector"]), ("Categoría", P["categoria"] + " - " + P["cargo"]), ("C.I.", P["ci"])]
    der = [("Funcionario", P["nro"]), ("Ingreso", fecha(*P["ingreso"])), ("Haberes del mes", f"{D['mes_nombre']}/ {D['anio']}")]
    for i, ((k1, v1), (k2, v2)) in enumerate(zip(izq, der)):
        lz.t(m, y + i * 22, f"{k1}: ", 15, True)
        lz.t(m + lz.ancho(f"{k1}: ", 15, True), y + i * 22, v1, 15)
        lz.t(W * .58, y + i * 22, f"{k2}: ", 15, True)
        lz.t(W * .58 + lz.ancho(f"{k2}: ", 15, True), y + i * 22, v2, 15)
    y += 80
    lz.t(m + 40, y, "Concepto", 16, True)
    lz.t(W * .3, y, "Haberes", 16, True)
    lz.t(W * .62, y, "Descuentos", 16, True)
    y += 30
    lz.l(0, y, W, y, 1, (150, 150, 150))
    y += 14
    es = rng.choice(["mono", "sans"])
    for h in L["H"]:
        lz.t(m + 40, y, h["concepto"][:24], 14, estilo=es)
        lz.t(W * .42, y, n(h["monto"]), 14, estilo=es, align="r")
        y += 23
    for d in L["D"]:
        lz.t(m + 40, y, d["concepto"][:24], 14, estilo=es)
        if d["tasa"]:
            lz.t(W * .62, y, f"{d['tasa'] * 100:.3f}%".replace(".", ","), 14, estilo=es, align="r")
            lz.t(W * .78, y, n(L["nominal"]), 14, estilo=es, align="r")
        lz.t(W - m, y, n(d["monto"]), 14, estilo=es, align="r")
        y += 23
    y += 10
    lz.l(0, y, W, y, 2)
    y += 16
    lz.t(W * .24, y, "Total Nominal:", 16, True, "r")
    lz.t(W * .42, y, n(L["total_h"]), 15, align="r", estilo=es)
    lz.t(W * .7, y, "Total Descuentos:", 16, True, "r")
    lz.t(W - m, y, n(L["total_d"]), 15, align="r", estilo=es)
    y += 40
    lz.t(W * .58, y + 6, "LÍQUIDO A COBRAR:", 17, True, "r")
    lz.r(W * .6, y, W - m, y + 36, 1)
    lz.t(W * .62, y + 8, "$", 17, True)
    lz.t(W - m - 10, y + 8, n(L["liquido"]), 17, True, "r")
    y += 60
    if y < H - 90:
        lz.parrafo(m, y, W * .4, "La empresa declara: Haber efectuado los aportes de seguridad social correspondientes a los haberes liquidados el mes anterior.", 11, estilo="mono", color=(110, 110, 110))
        lz.parrafo(W * .55, y, W * .4, "Recibí el importe mencionado y la copia correspondiente a la liquidación del mes de la fecha.", 11, estilo="mono", color=(110, 110, 110))
        lz.t(m, H - 40, f"Fecha de cobro: {D['fecha_pago']}", 11, estilo="mono")
        lz.t(W * .55, H - 40, "FIRMA ___________________________", 11, estilo="mono")
    return lz.img


def fam_periodos(rng):
    """Detalle de recibo con varios períodos apilados (tipo prestaciones/subsidios)."""
    W, H = 900, 1273
    lz, D = Lienzo(rng, W, H), datos(rng)
    E, P = D["emp"], D["per"]
    n = lambda v: fmt_num(v, D["fmt"])
    acc = elegir(rng, ACENTOS)
    m = 45
    y = 40
    logo(lz, m, y, 40, "".join(w[0] for w in ("Caja", "Previsión")), acc)
    org = rng.choice(["Caja de Prestaciones Sociales", "Fondo Social de Trabajadores", "Caja Notarial de Servicios", "Servicio de Haberes y Subsidios", E["nombre"]])
    lz.t(m, y + 44, org, 9, color=acc)
    lz.t(W * .45, y + 20, "Detalle del Recibo", 15)
    lz.l(W * .45, y + 40, W * .45 + lz.ancho("Detalle del Recibo", 15), y + 40, 1)
    lz.t(W - m, y, f"Recibo {rng.randint(100, 9999)}", 9, align="r")
    lz.t(W - m, y + 14, D["fecha_pago"], 9, True, "r")
    lz.t(W - m, y + 28, "Pág.nº    1", 9, True, "r")
    y += 80
    lz.l(m, y, W - m, y, 1)
    campos = [[("Emisión", f"{D['anio']}/{D['mes']:02d}"), ("Nro. de control", str(rng.randint(1000000, 99999999)))],
              None,
              [("Cédula", P["ci"])], [("Nombre completo", f"{P['nombres']} - {P['apellidos']}".upper())],
              None,
              [("Entidad pagadora", f"{rng.randint(1, 40)}   {P['banco']}")], [("Nivel 1", f"{rng.randint(100, 9999)}   {E['nombre']}")],
              [("Nivel 2", f"{rng.randint(1, 60)}   {E['ciudad'].upper()}")], None,
              [("Forma de pago", rng.choice(["Depósito", "Caja"])), ("Modalidad de pago", "Por Banco")], None,
              [("Domicilio", "")]]
    y += 8
    for fila in campos:
        if fila is None:
            y += 6
            lz.l(m, y, W - m, y, 1, (120, 120, 120))
            y += 6
            continue
        x = m
        for k, v in fila:
            lz.t(x, y, f"{k}:", 9, True)
            lz.t(x + 110, y, v, 9)
            x += 300
        y += 16
    y += 16
    lz.t(m, y, rng.choice(["Prestaciones", "Liquidaciones", "Haberes por período"]), 11, True)
    y += 18
    lz.t(m + 10, y, "Emisión  Rubro", 8, True)
    for t, xx in [("Haberes", .62), ("Descuentos", .72), ("Importe", .82), ("Vencim.", .93)]:
        lz.t(W * xx, y, t, 8, True, "r")
    y += 12
    lz.l(m, y, W - m, y, 1)
    y += 8
    rubro = rng.choice(["Subsidios por Enfermedad", "Haberes", "Subsidios por Maternidad", "Complemento", "Pasividad"])
    for b in range(rng.randint(4, 7)):
        if y > H - 120:
            break
        mm = (D["mes"] - b - 1) % 12 + 1
        lz.t(m + 10, y, f"{D['anio']}/{mm:02d}   {rng.randint(100, 999)}", 8)
        lz.t(m + 120, y, rubro, 8)
        lz.t(W * .33, y, f"Período Liq.: 01/{mm:02d}/{D['anio']} - 28/{mm:02d}/{D['anio']}", 7, color=(110, 110, 110))
        y += 12
        filas = []
        base = rng.uniform(1500, 30000)
        filas.append((str(rng.randint(1, 9)), rng.choice(["SALARIO VACACIONAL", "BÁSICO", "SUBSIDIO"]), base, 0))
        if rng.random() < .6:
            for nom, rate in [("S.N.I.S. 6%", .06), ("MONTEPÍO", .15), ("FRL", .00125)]:
                filas.append((str(rng.randint(10, 120)), nom, 0, base * rate))
        sub = 0
        for c, nom, hab, desc in filas:
            lz.t(m + 120, y, c, 7)
            lz.t(m + 150, y, nom, 7)
            if hab:
                lz.t(W * .62, y, n(hab), 7, align="r")
            if desc:
                lz.t(W * .72, y, n(desc), 7, align="r")
            sub += hab - desc
            y += 10
        lz.t(W * .74, y, "SubTotal:", 8, True, "r")
        lz.t(W * .82, y, n(sub), 8, align="r")
        y += 22
    return lz.img


def fam_tabla_densa(rng):
    """Recibo privado con tabla densa (cantidad, valor unitario, haberes, descuentos) y cajas al pie."""
    W, H = 1000, rng.randint(720, 860)
    lz, D = Lienzo(rng, W, H), datos(rng, rng.choice(["jornalero", "privado"]))
    E, P, L = D["emp"], D["per"], D["liq"]
    n = lambda v: fmt_num(v, D["fmt"])
    acc = elegir(rng, ACENTOS)
    m = 40
    y = 26
    logo(lz, m, y, 30, E["iniciales"], acc)
    lz.t(m + 38, y + 4, E["nombre"].split()[0].upper(), 20, True, color=acc)
    lz.t(W * .5, y, E["nombre"], 12, True, "c")
    lz.t(W * .5, y + 16, f"{E['direccion']}, {E['ciudad']}, URUGUAY", 10, False, "c")
    lz.t(W * .5, y + 32, f"RUT: {E['rut']}  BPS: {E['bps']}  MTSS: {E['mtss']}  Grupo: {P['grupo']}  BSE: {E['bse']}", 9, False, "c")
    lz.t(W - m, y, rng.choice(["Recibo de Haberes", "Recibo de Sueldo", "Liquidación de Haberes"]), 13, True, "r")
    y += 62
    lz.t(m + 10, y, f"{P['nro']} - {P['apellidos'].upper()} {P['nombres'].upper()}", 13, True)
    y += 22
    lz.r(m, y, W - m, y + 62, 1)
    info = [[("Categoría", P["categoria"]), ("F.Ingreso", fecha(*P["ingreso"])), ("C.I.", P["ci"])],
            [("Cargo", P["cargo"]), ("Valor Base", n(L["H"][0]["tasa"] or L["H"][0]["monto"] / 30))],
            [("Mes Liq.", f"{D['mes_nombre']} / {D['anio']}"), ("Descrip.Liq.", f"Liquidación {D['tipo']} {P['sector']} {D['mes']}/{D['anio']}")]]
    for i, fila in enumerate(info):
        for j, (k, v) in enumerate(fila):
            x = m + 10 + j * 300
            lz.t(x, y + 6 + i * 18, k, 10, True)
            lz.t(x + 80, y + 6 + i * 18, v, 10)
    y += 74
    cols = [("Concepto", .36, "l"), ("Cantidad", .11, "r"), ("V.Unitario", .12, "r"), ("Haberes", .14, "r"), ("Descuentos", .14, "r"), ("Rechazos", .13, "r")]
    filas = []
    for h in L["H"]:
        filas.append([f"{h['cod']}  {h['concepto']}", "" if h["cant"] is None else f"{h['cant']:.2f}", "" if h["tasa"] is None else n(h["tasa"]), n(h["monto"]), "", ""])
    for d in L["D"]:
        filas.append([f"{d['cod']}  {d['concepto']}", f"{(d['tasa'] or 0) * 100:.2f}" if d["tasa"] else "1.00", n(L["nominal"]) if d["tasa"] else "", "", n(d["monto"]), ""])
    y = tabla(lz, m + 20, y, W - 2 * m - 40, cols, filas, 15, 10, "none", None)
    y += 6
    lz.t(W * .52, y, "Subtotales", 11, True)
    lz.t(W * .74, y, n(L["total_h"]), 11, True, "r")
    lz.t(W * .87, y, n(L["total_d"]), 11, True, "r")
    lz.t(W * .97, y, n(0), 11, True, "r")
    y += 24
    lz.t(W * .52, y, "Líquido a Cobrar", 13, True)
    lz.t(W * .7, y, "$", 13, True)
    lz.t(W * .87, y, n(L["liquido"]), 13, True, "r")
    y += 30
    for k, v in [("Banco Empresa", P["banco"]), ("Banco Trabajador", P["banco"]), ("Fecha de pago", f"{D['fecha_pago']}      Nro. Cta. Func. {P['cuenta']}")]:
        lz.t(m + 10, y, f"{k}:", 9, True)
        lz.t(m + 110, y, v, 9)
        y += 14
    y += 8
    textos = ["La Empresa declara haber efectuado los aportes de seg. social correspondientes a los haberes liquidados del mes.",
              "FOCER: La empresa declara haber efectuado los aportes patronales y/o personales en el caso que correspondan.",
              "Recibí el importe total líquido y copia de esta liquidación, no teniendo nada que reclamar por ningún concepto."]
    bw = (W - 2 * m - 20) / 3
    for i, tx in enumerate(textos):
        x = m + i * (bw + 10)
        if y + 50 < H:
            lz.r(x, y, x + bw, y + 48, 1)
            lz.parrafo(x + 6, y + 5, bw - 12, tx, 8, 1.3)
    return lz.img


def fam_matriz(rng):
    """Copia impresa en impresora de matriz: tabla ASCII con '!' y '-', papel tintado, sello y firma."""
    W, H = 900, rng.randint(1000, 1200)
    papel = rng.choice([(246, 241, 214), (238, 242, 226), (244, 238, 222), (240, 240, 236)])
    tinta = rng.choice([(60, 60, 80), (40, 40, 40), (70, 60, 90)])
    lz, D = Lienzo(rng, W, H, papel, tinta), datos(rng, "publico")
    E, P, L = D["emp"], D["per"], D["liq"]
    n = lambda v: fmt_num(v, "uy")
    org = rng.choice(ORGANISMOS + [E["nombre"]])
    acc = rng.choice([(150, 40, 40), (40, 60, 140), (30, 100, 60)])
    logo(lz, 30, 30, 60, "".join(w[0] for w in org.split()[:2]), acc)
    lz.parrafo(30, 96, 120, org.upper(), 8, 1.2, color=acc, bold=True)
    s, es = 13, "mono"
    x0 = 165
    y = 40
    ancho_cols = 64

    def linea(txt):
        nonlocal y
        lz.t(x0, y, txt, s, estilo=es)
        y += s * 1.45

    linea(f"            COPIA LIQUIDACION DE SUELDO        +{'-' * 22}+")
    linea(f"            COBRA EN {P['banco'].upper()[:18]:<18}       ! Documento {rng.randint(1000, 9999)}.{rng.randint(100, 999)}-{rng.randint(1, 9)} !")
    linea(f"                                           +{'-' * 22}+")
    linea(f"! Programa {rng.randint(1, 9)}.{rng.randint(1, 30):02d}        Corresponde {D['mes_nombre']}/{D['anio']}")
    linea(f"                                    Ingreso {fecha(*P['ingreso'])}")
    linea(f"! Funcionario {P['nro']:>5} {(P['apellidos'] + ' ' + P['nombres']).upper()[:38]}")
    linea("+" + "-" * ancho_cols + "+")
    linea("! Código !     Concepto         !Cantidad!  Haberes  ! Descuento !")
    linea("+" + "-" * ancho_cols + "+")
    for h in L["H"]:
        c = "" if h["cant"] is None else f"{h['cant']:.2f}"
        linea(f"!{h['cod']:>6}  !{h['concepto'][:22]:<22}!{c:>8}!{n(h['monto']):>11}!           !")
    for d in L["D"]:
        linea(f"!{d['cod']:>6}  !{d['concepto'][:22]:<22}!        !           !{n(d['monto']):>11}!")
    for _ in range(rng.randint(0, 4)):
        linea("!        !                      !        !           !           !")
    linea("+" + "-" * ancho_cols + "+")
    linea(f"   GRADO {rng.randint(1, 16)}                     TOTALES  {n(L['total_h']):>11}  {n(L['total_d']):>11}")
    linea(f"   {P['cargo'].upper()[:28]:<28}       LIQUIDO  {n(L['liquido']):>11}     Red.-0,{rng.randint(10, 99)}")
    y += 30
    sello(lz, 200, y + 70, 70, ["", org.split()[0].upper(), ""], acc)
    firma(lz, W * .55, y + 40, 200, 70)
    lz.t(W * .55, y + 120, "Enc. Oficina Liquidaciones", 11, estilo=es, color=(40, 40, 120))
    # efecto matriz: leve desenfoque + desaturación
    img = lz.img.filter(ImageFilter.GaussianBlur(rng.uniform(.3, .7)))
    return img


def fam_listado_largo(rng):
    """Listado largo de conceptos con códigos (estilo organismo público/universidad)."""
    W, H = 900, 1273
    lz, D = Lienzo(rng, W, H), datos(rng, "publico")
    E, P, L = D["emp"], D["per"], D["liq"]
    n = lambda v: fmt_num(v, "uy")
    acc = elegir(rng, ACENTOS)
    org = rng.choice(ORGANISMOS)
    m = 40
    y = 30
    logo(lz, m, y, 44, "".join(w[0] for w in org.split()[:2]), acc)
    lz.t(m + 56, y + 4, org, 13, True)
    lz.t(m + 56, y + 22, rng.choice(["Sección Haberes", "División Personal", "Departamento de Sueldos"]), 11)
    lz.t(W * .62, y, "MES:", 10, True)
    lz.t(W * .7, y, f"{D['mes_nombre'].upper()} {D['anio']}", 10)
    lz.t(W * .62, y + 22, "Nº RECIBO:", 10, True)
    lz.t(W * .74, y + 22, str(rng.randint(1000, 99999)), 10)
    y += 64
    lz.r(m, y, W - m, y + 70, 1)
    for i, (k, v, k2, v2) in enumerate([("DIRECCIÓN:", E["direccion"], "RUT:", E["rut"]), ("BPS:", E["bps"], "BSE:", E["bse"]),
                                         ("FUNCIONARIO:", P["nro"], "DOCUMENTO:", P["ci"]), ("CARGO:", P["cargo"], "", "")]):
        lz.t(m + 8, y + 5 + i * 16, k, 9, True)
        lz.t(m + 100, y + 5 + i * 16, v, 9, color=(90, 90, 90))
        lz.t(W * .55, y + 5 + i * 16, k2, 9, True)
        lz.t(W * .66, y + 5 + i * 16, v2, 9, color=(90, 90, 90))
    y += 80
    lz.t(m + 8, y, "CONCEPTO  DESCRIPCIÓN", 10, True)
    lz.t(W * .72, y, "HABERES", 10, True, "r")
    lz.t(W - m - 8, y, "DESCUENTOS", 10, True, "r")
    y += 16
    lz.l(m, y, W - m, y, 1)
    y += 4
    extra_desc = ["DEC.06/06 HD RECUP.SALARIAL", "AUM.MAYO/03 TAS", "LEY 20075 Res.8 CDC", "DEC.22/007 HD RECUP.SALARIAL",
                  "ANTIGÜEDAD TAS", "DEC.41/008 HD RECUP.SALARIAL", "INCR.HD ENERO 2008", "DEC.19/01/09 HD RECUP",
                  "CUOTA ALIMENTACION", "DED.COMPENSADA TAS", "AUM.5/92", "RECUP.SALARIAL 2010 HD", "HOGAR CONSTITUIDO",
                  "AUM.1/94 TAS", "COMPENS. DEDICACIÓN", "PRIMA TÍTULO", "AJUSTE IPC", "DIF. GRADO"]
    rng.shuffle(extra_desc)
    filas_h = [(h["cod"], h["concepto"], h["monto"]) for h in L["H"]]
    for i, desc in enumerate(extra_desc[: rng.randint(6, 14)]):
        filas_h.append((str(rng.randint(10, 400)), desc, rng.uniform(150, 9000)))
    filas_h.sort(key=lambda t: int(t[0]) if t[0].isdigit() else 0)
    zebra = rng.random() < .6
    for i, (c, desc, monto) in enumerate(filas_h):
        if zebra and i % 2 == 0:
            lz.d.rectangle([m + 2, y - 1, W - m - 2, y + 14], fill=(238, 238, 238))
        lz.t(m + 30, y, c, 9, align="r", color=(80, 80, 80))
        lz.t(m + 40, y, desc, 9, color=(80, 80, 80))
        lz.t(W * .72, y, n(monto), 9, align="r", color=(80, 80, 80))
        y += 16
    y += 12
    tot_d = 0
    for d in L["D"]:
        lz.t(m + 30, y, str(rng.randint(1000, 99999)), 9, align="r", color=(80, 80, 80))
        lz.t(m + 40, y, d["concepto"], 9, color=(80, 80, 80))
        lz.t(W - m - 8, y, n(d["monto"]), 9, align="r", color=(80, 80, 80))
        tot_d += d["monto"]
        y += 16
    tot_h = sum(f[2] for f in filas_h)
    y += 10
    lz.t(W * .55, y, "SUBTOTALES:", 10, True, "r")
    lz.t(W * .72, y, n(tot_h), 10, align="r")
    lz.t(W - m - 8, y, n(tot_d), 10, align="r")
    y += 18
    lz.t(W * .55, y, "LÍQUIDO:", 10, True, "r")
    lz.t(W * .72, y, n(tot_h - tot_d), 10, True, "r")
    y += 30
    lz.l(m, y, W - m, y, 1)
    y += 8
    lz.t(m + 8, y, "MONTO GRAVADO IRPF:", 9, True)
    lz.t(m + 170, y, n(tot_h), 9)
    lz.t(W * .5, y, "DEDUCCIONES IRPF:", 9, True)
    lz.t(W * .68, y, n(tot_h * .17), 9)
    lz.t(m + 8, y + 16, "MONTO COMPUTADO IRPF:", 9, True)
    lz.t(m + 170, y + 16, n(tot_h * 1.06), 9)
    lz.t(m + 8, y + 50, "ORDEN ENTREGA:", 9, True)
    return lz.img


def fam_lado_a_lado(rng):
    """Haberes y descuentos en dos tablas lado a lado, datos del empleado en grilla chica."""
    W, H = 1000, rng.randint(700, 880)
    lz, D = Lienzo(rng, W, H), datos(rng)
    E, P, L = D["emp"], D["per"], D["liq"]
    n = lambda v: fmt_num(v, D["fmt"])
    acc = elegir(rng, ACENTOS)
    m = 30
    y = 24
    lz.t(m, y, E["nombre"].upper(), 14, True)
    lz.t(m, y + 20, f"Dirección: {E['direccion']}", 8)
    lz.t(m, y + 32, f"RUT: {E['rut']}", 8)
    lz.t(W * .45, y + 4, f"BPS: {E['bps']}", 8)
    lz.t(W * .45, y + 16, f"BSE: {E['bse']}", 8)
    lz.t(W * .45, y + 28, f"Grupo / Sub: {P['grupo']} / {rng.randint(1, 9)}", 8)
    lz.t(W - m, y, f"{D['mes_nombre']} {D['anio']}", 14, False, "r")
    lz.t(W - m, y + 24, f"Liquidación:   {D['tipo']}", 9, False, "r")
    if rng.random() < .5:
        lz.t(W - m, y + 40, f"Recibo de Sueldo Número: {rng.randint(1000, 999999)}", 9, False, "r")
    y += 58
    lz.l(m, y, W - m, y, 2, acc)
    lz.t(m + 10, y + 4, "DATOS DEL EMPLEADO", 8, True)
    y += 18
    grid = [("Nombres", P["nombres"]), ("Sector", P["sector"]), ("Banco", P["banco"]), ("Apellidos", P["apellidos"]),
            ("Cargo", P["cargo"]), ("Nº Cuenta", P["cuenta"]), ("C.I.", P["ci"]), ("Remuneración", "Mensual"),
            ("Fecha de ingreso", fecha(*P["ingreso"])), ("Nº Func.", P["nro"]), ("Horario", "Lunes a viernes 9 a 18"), ("Sist. de Retiro", "BPS")]
    for i, (k, v) in enumerate(grid):
        col, fila = i % 3, i // 3
        x = m + col * (W - 2 * m) / 3
        lz.t(x, y + fila * 13, f"{k}:", 8, True)
        lz.t(x + 85, y + fila * 13, v, 8)
    y += 4 * 13 + 14
    half = (W - 2 * m - 20) / 2
    lz.d.rectangle([m, y, m + half, y + 16], fill=acc)
    lz.d.rectangle([m + half + 20, y, W - m, y + 16], fill=acc)
    lz.t(m + 6, y + 2, "HABERES", 10, True, color=(255, 255, 255))
    lz.t(m + half + 26, y + 2, "DESCUENTOS", 10, True, color=(255, 255, 255))
    y += 20
    ch = [("Nombre", .45, "l"), ("Detalle", .3, "l"), ("Importe", .25, "r")]
    fh = [[h["concepto"], (f"{h['cant']} x {n(h['tasa'])}" if h["tasa"] else (f"{h['cant']} x {n(h['monto'] / max(h['cant'] or 1, 1))}" if h["cant"] else "")), n(h["monto"])] for h in L["H"]]
    fd = [[d["concepto"], (f"{d['tasa'] * 100:.2f}% de {n(L['nominal'])}" if d["tasa"] else ""), n(d["monto"])] for d in L["D"]]
    alto = max(len(fh), len(fd)) * 15 + 20
    y1 = tabla(lz, m, y, half, ch, fh, 15, 9, "header", None, alto_min=alto, color_borde=acc)
    tabla(lz, m + half + 20, y, half, ch, fd, 15, 9, "header", None, alto_min=alto, color_borde=acc)
    y = y1 + 6
    lz.l(m, y, W - m, y, 1, acc)
    lz.t(m, y + 4, f"Total Gravado: {n(L['nominal'])}", 9)
    lz.t(m + half * .55, y + 4, f"Total de Haberes:  {n(L['total_h'])}", 9, True)
    lz.t(W - m, y + 4, f"Total de Descuentos:  {n(L['total_d'])}", 9, True, "r")
    y += 30
    lz.parrafo(m, y, half, "RECIBO DE CONTROL!!! POR FAVOR EN CASO DE TENER DIFERENCIAS COMUNICARSE CON RRHH. La empresa declara haber efectuado los aportes de seguridad social correspondientes a los haberes del mes anterior.", 8)
    lz.t(W - m, y + 10, "Líquido a Cobrar :", 11, True, "r")
    lz.t(W - m, y + 30, f"$ {n(L['liquido'])}", 13, True, "r")
    lz.l(W - m - 200, y + 80, W - m, y + 80, 1)
    lz.t(W - m - 100, y + 84, "Firma", 8, False, "c")
    if rng.random() < .5:
        firma(lz, W - m - 190, y + 40, 170, 40)
    return lz.img


def fam_formulario(rng, completo=True):
    """Formulario impreso 'Recibo de Sueldos y Jornales' completado a mano (tinta azul).
    Con completo=False sale VACÍO: eso es un NEGATIVO (recibo sin carácter válido)."""
    W, H = 900, 1273
    lz, D = Lienzo(rng, W, H), datos(rng)
    E, P, L = D["emp"], D["per"], D["liq"]
    n = lambda v: fmt_num(v, D["fmt"])
    tinta_mano = rng.choice([(20, 40, 150), (10, 10, 60), (30, 30, 30)])
    estilo = rng.choice(["bandas", "planilla"])
    banda = rng.choice([(70, 70, 70), (60, 60, 110), (40, 40, 40)]) if estilo == "bandas" else rng.choice([(200, 190, 230), (210, 225, 240), (220, 220, 220)])
    m = 60
    y = 40

    def mano(x, yy, txt, size=13):
        if completo and txt:
            lz.t(x, yy, txt, size, False, estilo="sans", color=tinta_mano, italic=True)

    if estilo == "bandas":
        lz.t(m, y, rng.choice(["Recibos de Sueldos y Jornales", "Recibo de Sueldo"]), 18)
        y += 28
    else:
        lz.r(m, y, W - m, y + 24, 1, fill=banda)
        lz.t(W / 2, y + 4, "RECIBO DE SUELDO Y JORNALES", 15, True, "c")
        y += 24

    def seccion(tit):
        nonlocal y
        if estilo == "bandas":
            lz.d.rectangle([m, y, W - m, y + 18], fill=banda)
            lz.t(W / 2, y + 2, tit, 12, True, "c", color=(255, 255, 255))
        else:
            lz.r(m, y, W - m, y + 18, 1, fill=banda)
            lz.t(W / 2, y + 2, tit, 11, True, "c")
        y += 22

    def campo(etq, val="", x=None, ancho=None, size=11):
        x = x if x is not None else m + 8
        lz.t(x, y, etq, size)
        x2 = x + lz.ancho(etq, size) + 6
        fin = (x + ancho) if ancho else W - m - 8
        lz.punteada(x2, y + size + 2, fin, 3, (120, 120, 120)) if estilo == "bandas" else lz.l(m, y + size + 5, W - m, y + size + 5, 1, (160, 160, 160))
        mano(x2 + 6, y - 3, val)

    seccion("EMPRESA")
    for etq, val in [("Razón social:", E["nombre"]), ("Domicilio:", E["direccion"]), ("B.P.S. Nº:", E["bps"]), ("R.U.T./C.I.:", E["rut"])]:
        campo(etq, val)
        y += 22
    seccion("EMPLEADO")
    for etq, val in [("Apellido, nombre:", f"{P['apellidos']}, {P['nombres']}"), ("Cargo Laboral:", P["cargo"]),
                     ("Fecha de ingreso:", fecha(*P["ingreso"])), ("C.I.:", P["ci"]), ("Jornal / Mensual:", "Mensual"),
                     ("Liquidación de pago mes:", f"{D['mes_nombre']} de {D['anio']}")]:
        campo(etq, val)
        y += 22
    seccion("NOMINAL")
    rubros_h = ["Sueldo", "Horas extras", "Horas nocturnas", "Feriados pagos", "Aguinaldo", "Licencia", "Salario vacacional", "Indemnización"]
    claves_h = {"Horas extras": ["extra"], "Horas nocturnas": ["noctur"], "Feriados pagos": ["feriado"], "Aguinaldo": ["aguinaldo"],
                "Licencia": ["licencia"], "Salario vacacional": ["vacacional"], "Indemnización": ["indemn"]}
    montos_h = {r: 0.0 for r in rubros_h}
    for h in L["H"]:
        c = h["concepto"].lower()
        destino = next((r for r, ks in claves_h.items() if any(k in c for k in ks)), "Sueldo")
        montos_h[destino] += h["monto"]
    valores = {r: (n(v) if v else "") for r, v in montos_h.items()}
    xm = W - m - 170
    for r in rubros_h:
        lz.r(m, y, W - m, y + 22, 1, (150, 150, 150))
        lz.l(xm, y, xm, y + 22, 1, (150, 150, 150))
        lz.t(m + 8, y + 4, r, 11)
        lz.t(xm + 6, y + 4, "$", 11)
        mano(xm + 30, y + 2, valores[r])
        y += 22
    lz.t(xm - 10, y + 6, "TOTAL GANADO NOMINAL", 11, True, "r")
    lz.r(xm, y + 2, W - m, y + 24, 1)
    lz.t(xm + 6, y + 6, "$", 11)
    mano(xm + 30, y + 4, n(sum(montos_h.values())))
    y += 34
    seccion("DESCUENTOS")
    rubros_d = ["Aporte jubilatorio", "Seguro de enfermedad", "I.R.P.F.", "F.R.L.", "Adelantos"]
    claves_d = {"Aporte jubilatorio": ["jub", "montep"], "Seguro de enfermedad": ["fonasa", "snis", "enfermedad"],
                "I.R.P.F.": ["irpf", "i.r.p.f"], "F.R.L.": ["frl", "f.r.l", "fondo rec"], "Adelantos": ["adelanto"]}
    montos_d = {r: 0.0 for r in rubros_d}
    for dd in L["D"]:
        c = dd["concepto"].lower()
        destino = next((r for r, ks in claves_d.items() if any(k in c for k in ks)), None)
        if destino:
            montos_d[destino] += dd["monto"]
    for r in rubros_d:
        lz.r(m, y, W - m, y + 22, 1, (150, 150, 150))
        lz.l(xm, y, xm, y + 22, 1, (150, 150, 150))
        lz.t(m + 8, y + 4, r, 11)
        lz.t(xm + 6, y + 4, "$", 11)
        if montos_d[r]:
            mano(xm + 30, y + 2, n(montos_d[r]))
        y += 22
    lz.t(xm - 10, y + 6, "TOTAL DESCUENTOS", 11, True, "r")
    lz.r(xm, y + 2, W - m, y + 24, 1)
    lz.t(xm + 6, y + 6, "$", 11)
    mano(xm + 30, y + 4, n(sum(montos_d.values())))
    y += 30
    lz.t(xm - 10, y + 6, "CANTIDAD A COBRAR", 12, True, "r")
    lz.r(xm, y + 2, W - m, y + 26, 2)
    lz.t(xm + 6, y + 6, "$", 11)
    mano(xm + 30, y + 4, n(sum(montos_h.values()) - sum(montos_d.values())), 14)
    y += 46
    lz.t(m + 8, y, "_________________ de ________ de 20____", 11)
    lz.parrafo(W * .55, y - 4, W * .38, "Recibí el importe y una copia de esta liquidación no teniendo nada que reclamar por ningún concepto.", 10)
    y += 50
    for tx in ["Declaro haber efectuado los aportes correspondientes a los haberes liquidados en el mes anterior.",
               "Declaro no haber efectuado los aportes correspondientes por haber vencido, en carácter de aplazamiento."]:
        lz.parrafo(m + 8, y, W * .6, tx, 9)
        lz.r(W * .72, y, W * .72 + 16, y + 16, 1)
        y += 34
    if completo and rng.random() < .7:
        firma(lz, W * .6, y - 10, 180, 50, tinta_mano)
    return lz.img


FAMILIAS = {
    "bloques": fam_bloques,
    "sistema_viejo": fam_sistema_viejo,
    "generico_limpio": fam_generico_limpio,
    "periodos": fam_periodos,
    "tabla_densa": fam_tabla_densa,
    "matriz": fam_matriz,
    "listado_largo": fam_listado_largo,
    "lado_a_lado": fam_lado_a_lado,
    "formulario": fam_formulario,
}

# ---------------------------------------------------------------------------
# NEGATIVOS (documentos que NO son recibos de sueldo)
# ---------------------------------------------------------------------------
FRASES = [
    "Por medio de la presente nos dirigimos a ustedes a efectos de informar las novedades del período.",
    "Les recordamos que el plazo para la presentación de la documentación vence el próximo día hábil.",
    "Quedamos a las órdenes por cualquier consulta que deseen realizar.",
    "La reunión de coordinación se realizará en la sala de conferencias del segundo piso.",
    "Se solicita a todos los sectores remitir el informe mensual antes del cierre.",
    "Agradecemos desde ya la atención prestada y la colaboración de siempre.",
    "En virtud de lo expuesto, se resuelve aprobar la propuesta presentada por la comisión.",
    "Cabe destacar que los trabajos de mantenimiento se extenderán durante toda la semana.",
    "Se adjunta a la presente el detalle de los materiales entregados en la última visita.",
    "Solicitamos confirmar la recepción de esta nota a la brevedad posible.",
    "La nueva modalidad de atención al público comenzará a regir a partir del mes próximo.",
    "Les informamos que el horario de oficina será de lunes a viernes de 9 a 17 horas.",
    "Ante cualquier duda, pueden comunicarse con nuestro departamento comercial.",
    "El comité evaluará las propuestas recibidas y comunicará la resolución por escrito.",
    "Se recuerda la importancia de mantener actualizados los datos de contacto.",
    "El inventario anual se llevará a cabo durante la última semana del mes.",
    "Con motivo de las obras en la vía pública, el acceso se realizará por la puerta lateral.",
    "La presente propuesta tiene una validez de treinta días a partir de su emisión.",
    "Se informa que los pedidos recibidos después de las 15 horas se despacharán al día siguiente.",
    "Los interesados deberán presentarse con la documentación original y una copia.",
]
PRODUCTOS = ["Resma papel A4", "Tóner impresora", "Cable UTP Cat6 (m)", "Tornillo 8x40 (caja)", "Pintura látex 20L",
             "Servicio de instalación", "Mano de obra", "Lámpara LED 12W", "Silla de oficina", "Disco SSD 1TB",
             "Leche entera 1L", "Pan flauta", "Arroz 1kg", "Yerba 1kg", "Aceite girasol 900ml", "Café molido 250g",
             "Detergente 750ml", "Agua mineral 2L", "Fideos 500g", "Queso dambo kg", "Jamón cocido kg",
             "Bolsa de portland", "Arena (m3)", "Caño PVC 110", "Flete Montevideo", "Horas de consultoría",
             "Licencia software anual", "Mantenimiento preventivo", "Repuesto bomba", "Neumático 175/70 R13"]


SUPER = ["Leche entera 1L", "Pan flauta", "Arroz 1kg", "Yerba 1kg", "Aceite girasol 900ml", "Café molido 250g",
         "Detergente 750ml", "Agua mineral 2L", "Fideos 500g", "Queso dambo kg", "Jamón cocido kg", "Manzana kg",
         "Banana kg", "Papa kg", "Huevos x12", "Azúcar 1kg", "Harina 000 1kg", "Galletitas", "Dulce de leche",
         "Papel higiénico x4", "Jabón tocador", "Refresco 1,5L", "Cerveza 1L", "Tomate kg", "Cebolla kg"]


def _items(rng, k, pmin=40, pmax=12000):
    out = []
    for _ in range(k):
        cant = rng.choice([1, 1, 1, 2, 3, 4, 5, 10, 12, round(rng.uniform(.3, 3), 3)])
        pu = round(rng.uniform(pmin, pmax), 2)
        out.append((elegir(rng, PRODUCTOS), cant, pu, round(cant * pu, 2)))
    return out


def _qr(lz, x, y, s):
    rng = lz.rng
    k = 21
    c = s / k
    for i in range(k):
        for j in range(k):
            if rng.random() < .45:
                lz.d.rectangle([x + i * c, y + j * c, x + (i + 1) * c, y + (j + 1) * c], fill=(0, 0, 0))
    for (a, b) in [(0, 0), (k - 7, 0), (0, k - 7)]:
        lz.d.rectangle([x + a * c, y + b * c, x + (a + 7) * c, y + (b + 7) * c], fill=(255, 255, 255))
        lz.r(x + a * c, y + b * c, x + (a + 7) * c, y + (b + 7) * c, int(c), (0, 0, 0))
        lz.d.rectangle([x + (a + 2) * c, y + (b + 2) * c, x + (a + 5) * c, y + (b + 5) * c], fill=(0, 0, 0))


def _barcode(lz, x, y, w, h):
    rng = lz.rng
    xx = x
    while xx < x + w:
        bw = rng.choice([1, 1, 2, 3])
        if rng.random() < .55:
            lz.d.rectangle([xx, y, xx + bw - 1, y + h], fill=(0, 0, 0))
        xx += bw


def neg_factura(rng):
    W, H = 900, 1273
    lz = Lienzo(rng, W, H)
    E, C = empresa(rng), empresa(rng)
    fm = rng.choice(["uy", "uy", "en"])
    n = lambda v: fmt_num(v, fm)
    acc = elegir(rng, ACENTOS)
    m = 45
    y = 40
    logo(lz, m, y, 54, E["iniciales"], acc)
    lz.t(m + 66, y + 2, E["nombre"], 17, True)
    lz.t(m + 66, y + 24, f"{E['direccion']} - {E['ciudad']}", 11)
    lz.t(m + 66, y + 40, f"Tel. {E['tel']}", 11)
    bx = W * .58
    lz.r(bx, y - 5, W - m, y + 95, 1)
    lz.t(bx + 12, y + 2, f"RUT EMISOR: {E['rut']}", 11, True)
    lz.t(bx + 12, y + 22, rng.choice(["e-Factura", "e-Ticket", "e-Factura Crédito", "FACTURA"]), 16, True)
    lz.t(bx + 12, y + 46, f"Serie {rng.choice('ABCD')}   Nº {rng.randint(1000, 999999)}", 12)
    lz.t(bx + 12, y + 66, f"Fecha: {fecha(rng.randint(1, 28), rng.randint(1, 12), rng.randint(2019, 2026))}", 12)
    y += 120
    lz.r(m, y, W - m, y + 72, 1)
    lz.t(m + 10, y + 8, "CLIENTE:", 11, True)
    lz.t(m + 90, y + 8, C["nombre"], 11)
    lz.t(m + 10, y + 28, "RUT:", 11, True)
    lz.t(m + 90, y + 28, C["rut"], 11)
    lz.t(m + 10, y + 48, "Dirección:", 11, True)
    lz.t(m + 90, y + 48, f"{C['direccion']}, {C['ciudad']}", 11)
    lz.t(W * .6, y + 8, f"Moneda: {rng.choice(['UYU', 'USD'])}", 11)
    lz.t(W * .6, y + 28, f"Forma de pago: {rng.choice(['Contado', 'Crédito 30 días'])}", 11)
    y += 90
    its = _items(rng, rng.randint(3, 14))
    cols = [("Cant.", .1, "r"), ("Descripción", .5, "l"), ("P. Unitario", .2, "r"), ("Importe", .2, "r")]
    filas = [[f"{c:g}", d, n(pu), n(t)] for d, c, pu, t in its]
    y = tabla(lz, m, y, W - 2 * m, cols, filas, 20, 11, rng.choice(["full", "header"]), rng.choice([None, (235, 235, 235)]), alto_min=rng.randint(250, 520), color_borde=acc)
    sub = sum(t for *_, t in its)
    iva = sub * rng.choice([.22, .10])
    y += 16
    for k, v, b in [("Subtotal", sub, False), (f"IVA {'22' if iva / sub > .15 else '10'}%", iva, False), ("TOTAL", sub + iva, True)]:
        lz.t(W * .72, y, k, 13 if b else 12, b, "r")
        lz.t(W - m, y, n(v), 13 if b else 12, b, "r")
        y += 24
    y += 20
    _qr(lz, m, y, 110)
    lz.t(m + 130, y + 10, f"CAE Nº {rng.randint(10 ** 9, 10 ** 10)}  Serie {rng.choice('ABCD')}", 10)
    lz.t(m + 130, y + 28, f"Vencimiento CAE: {fecha(rng.randint(1, 28), rng.randint(1, 12), rng.randint(2025, 2028))}", 10)
    lz.t(m + 130, y + 46, f"Código de seguridad: {''.join(rng.choice('ABCDEF0123456789') for _ in range(6))}", 10)
    lz.t(m + 130, y + 64, "Puede verificar comprobante en el sitio web de la autoridad tributaria", 9)
    return lz.img


def neg_estado_cuenta(rng):
    W, H = 900, 1273
    lz = Lienzo(rng, W, H)
    P = persona(rng)
    banco = elegir(rng, BANCOS)
    fm = rng.choice(["uy", "en"])
    n = lambda v: fmt_num(v, fm)
    acc = elegir(rng, ACENTOS)
    m = 45
    y = 40
    logo(lz, m, y, 46, "".join(w[0] for w in banco.split()[:2]), acc)
    lz.t(m + 58, y + 4, banco, 18, True, color=acc)
    lz.t(W - m, y + 6, "ESTADO DE CUENTA", 15, True, "r")
    y += 70
    anio, mes = rng.randint(2019, 2026), rng.randint(1, 12)
    for k, v in [("Titular", f"{P['nombres']} {P['apellidos']}".upper()), ("Cuenta", f"Caja de Ahorro {rng.choice(['$', 'U$S'])} {P['cuenta']}"),
                 ("Período", f"01/{mes:02d}/{anio} al 30/{mes:02d}/{anio}"), ("Sucursal", f"{rng.randint(1, 99):02d} - {elegir(rng, CIUDADES)}")]:
        lz.t(m, y, f"{k}:", 11, True)
        lz.t(m + 90, y, v, 11)
        y += 18
    y += 16
    saldo = rng.uniform(2000, 90000)
    filas = [[f"01/{mes:02d}", "SALDO ANTERIOR", "", "", n(saldo)]]
    conceptos = ["COMPRA TARJETA DÉBITO", "TRANSFERENCIA RECIBIDA", "TRANSFERENCIA ENVIADA", "RETIRO CAJERO",
                 "PAGO SERVICIOS", "DÉBITO AUTOMÁTICO", "COMISIÓN MANTENIMIENTO", "DEPÓSITO", "PAGO TARJETA CRÉDITO",
                 "ACRED. SUELDOS " + nombre_empresa(rng).upper()[:22], "COMPRA WEB", "INTERESES"]
    dia = 1
    hubo_sueldo = False
    for _ in range(rng.randint(14, 32)):
        dia = min(30, dia + rng.randint(0, 2))
        c = elegir(rng, conceptos)
        while "SUELDOS" in c and hubo_sueldo:
            c = elegir(rng, conceptos)
        hubo_sueldo = hubo_sueldo or "SUELDOS" in c
        credito = any(w in c for w in ("RECIBIDA", "DEPÓSITO", "SUELDOS", "INTERESES"))
        mto = rng.uniform(25000, 90000) if "SUELDOS" in c else rng.uniform(40, max(500, min(15000, saldo * .4)) if not credito else 15000)
        saldo += mto if credito else -mto
        filas.append([f"{dia:02d}/{mes:02d}", c, "" if credito else n(mto), n(mto) if credito else "", n(saldo)])
    cols = [("Fecha", .1, "l"), ("Concepto", .45, "l"), ("Débito", .15, "r"), ("Crédito", .15, "r"), ("Saldo", .15, "r")]
    y = tabla(lz, m, y, W - 2 * m, cols, filas, 18, 10, rng.choice(["header", "full"]), (230, 235, 240), zebra=(245, 247, 250) if rng.random() < .5 else None, color_borde=acc)
    y += 20
    lz.t(W - m, y, f"SALDO FINAL: {n(saldo)}", 12, True, "r")
    lz.parrafo(m, H - 90, W - 2 * m, "Por consultas comuníquese con nuestro centro de atención telefónica. Verifique los movimientos de su cuenta y comunique cualquier diferencia dentro de los 30 días.", 9, color=(110, 110, 110))
    return lz.img


def neg_ticket(rng):
    W, H = 420, rng.randint(900, 1250)
    lz = Lienzo(rng, W, H, papel=rng.choice([(250, 250, 248), (245, 245, 240), (255, 255, 255)]))
    E = empresa(rng)
    n = lambda v: fmt_num(v, "uy")
    s, es = 12, "mono"
    y = 20
    for txt in [E["nombre"].upper()[:30], E["direccion"][:30], f"RUT {E['rut']}", "e-Ticket", f"Nº {rng.randint(10000, 9999999)}",
                fecha(rng.randint(1, 28), rng.randint(1, 12), rng.randint(2019, 2026)) + f"  {rng.randint(8, 21):02d}:{rng.randint(0, 59):02d}"]:
        lz.t(W / 2, y, txt, s, txt.startswith(E["nombre"][:3].upper()), "c", es)
        y += s * 1.5
    lz.t(20, y, "-" * 40, s, estilo=es)
    y += s * 1.5
    its = [(elegir(rng, SUPER), c, pu, t) for _, c, pu, t in _items(rng, rng.randint(5, 22), 30, 600)]
    for d, c, pu, t in its:
        lz.t(20, y, f"{c:g} x {n(pu)}", s - 1, estilo=es)
        y += s * 1.3
        lz.t(20, y, d[:22], s - 1, estilo=es)
        lz.t(W - 20, y, n(t), s - 1, align="r", estilo=es)
        y += s * 1.5
        if y > H - 200:
            break
    lz.t(20, y, "-" * 40, s, estilo=es)
    y += s * 1.6
    tot = sum(t for *_, t in its)
    lz.t(20, y, "TOTAL", s + 3, True, estilo=es)
    lz.t(W - 20, y, n(tot), s + 3, True, "r", es)
    y += s * 2
    lz.t(20, y, f"IVA incluido: {n(tot * .18)}", s - 1, estilo=es)
    y += s * 1.5
    lz.t(20, y, f"Pago: {rng.choice(['EFECTIVO', 'DÉBITO', 'CRÉDITO'])}", s - 1, estilo=es)
    y += s * 2
    _qr(lz, W / 2 - 50, y, 100)
    y += 115
    lz.t(W / 2, y, "GRACIAS POR SU COMPRA", s, True, "c", es)
    return lz.img


def neg_presupuesto(rng):
    W, H = 900, 1273
    lz = Lienzo(rng, W, H)
    E, C = empresa(rng), empresa(rng)
    n = lambda v: fmt_num(v, rng.choice(["uy", "en"]))
    acc = elegir(rng, ACENTOS)
    m = 55
    y = 45
    lz.d.rectangle([0, 0, W, 10], fill=acc)
    lz.t(m, y, E["nombre"], 20, True, color=acc)
    lz.t(m, y + 28, f"{E['direccion']} · {E['ciudad']} · Tel. {E['tel']}", 10)
    lz.t(W - m, y, rng.choice(["PRESUPUESTO", "COTIZACIÓN", "PROPUESTA COMERCIAL"]), 18, True, "r")
    lz.t(W - m, y + 28, f"Nº {rng.randint(100, 9999)} - {fecha(rng.randint(1, 28), rng.randint(1, 12), rng.randint(2019, 2026))}", 11, False, "r")
    y += 80
    lz.t(m, y, "Sres.:", 12, True)
    lz.t(m + 60, y, C["nombre"], 12)
    lz.t(m, y + 20, "Atención:", 12, True)
    lz.t(m + 80, y + 20, f"{elegir(rng, NOMBRES)} {elegir(rng, APELLIDOS)}", 12)
    y += 54
    y = lz.parrafo(m, y, W - 2 * m, " ".join(rng.sample(FRASES, 2)), 12) + 10
    its = _items(rng, rng.randint(3, 10))
    cols = [("Ítem", .08, "l"), ("Descripción", .5, "l"), ("Cant.", .1, "r"), ("Precio", .16, "r"), ("Total", .16, "r")]
    y = tabla(lz, m, y, W - 2 * m, cols, [[str(i + 1), d, f"{c:g}", n(pu), n(t)] for i, (d, c, pu, t) in enumerate(its)], 22, 11, "full",
              (235, 235, 235), color_borde=acc)
    tot = sum(t for *_, t in its)
    y += 14
    lz.t(W - m - 160, y, "TOTAL (IVA no incluido)", 12, True, "r")
    lz.t(W - m, y, n(tot), 12, True, "r")
    y += 50
    lz.t(m, y, "Condiciones:", 12, True)
    y += 20
    for c in ["Validez de la oferta: 15 días.", f"Forma de pago: {rng.choice(['50% anticipo, saldo contra entrega', 'contado', '30 días fecha factura'])}.",
              f"Plazo de entrega: {rng.randint(2, 30)} días hábiles."]:
        lz.t(m + 10, y, "• " + c, 11)
        y += 18
    y += 30
    firma(lz, W - m - 220, y, 180, 50)
    lz.t(W - m - 130, y + 60, "Departamento Comercial", 10, False, "c")
    return lz.img


def neg_constancia(rng):
    """Constancia laboral: habla de salario pero NO es un recibo (negativo difícil)."""
    W, H = 900, 1273
    lz = Lienzo(rng, W, H)
    E, P = empresa(rng), persona(rng)
    acc = elegir(rng, ACENTOS)
    m = 90
    y = 60
    logo(lz, m, y, 50, E["iniciales"], acc)
    lz.t(m + 62, y + 6, E["nombre"], 16, True, color=acc)
    lz.t(m + 62, y + 28, f"{E['direccion']} - {E['ciudad']} - RUT {E['rut']}", 10)
    y += 120
    lz.t(W - m, y, f"{E['ciudad']}, {rng.randint(1, 28)} de {elegir(rng, MESES).lower()} de {rng.randint(2019, 2026)}", 12, False, "r")
    y += 60
    lz.t(W / 2, y, rng.choice(["CONSTANCIA", "CONSTANCIA DE TRABAJO", "CERTIFICADO LABORAL"]), 18, True, "c")
    y += 60
    sal = rng.uniform(25000, 160000)
    texto = (f"Por la presente se deja constancia que {rng.choice(['el Sr.', 'la Sra.'])} {P['nombres']} {P['apellidos']}, "
             f"titular de la cédula de identidad Nº {P['ci']}, se desempeña en esta empresa desde el {fecha(*P['ingreso'])} "
             f"en el cargo de {P['cargo']}, con una carga horaria de {rng.choice([20, 30, 40, 44, 48])} horas semanales, "
             f"percibiendo un salario nominal mensual de $ {fmt_num(sal, 'uy')}.")
    estilo = rng.choice(["serif", "sans"])
    y = lz.parrafo(m, y, W - 2 * m, texto, 14, 1.7, estilo) + 20
    y = lz.parrafo(m, y, W - 2 * m, "Se expide la presente a solicitud del interesado para ser presentada ante quien corresponda.", 14, 1.7, estilo) + 100
    firma(lz, W / 2 - 90, y, 180, 60)
    y += 70
    lz.l(W / 2 - 140, y, W / 2 + 140, y, 1)
    lz.t(W / 2, y + 6, f"{elegir(rng, NOMBRES)} {elegir(rng, APELLIDOS)}", 12, False, "c")
    lz.t(W / 2, y + 24, rng.choice(["Gerente de Recursos Humanos", "Director", "Encargado de Personal"]), 11, False, "c")
    if rng.random() < .5:
        sello(lz, W / 2 + 200, y - 20, 60, ["", E["iniciales"], ""], acc)
    return lz.img


def neg_factura_servicios(rng):
    W, H = 900, 1273
    lz = Lienzo(rng, W, H)
    P = persona(rng)
    acc = elegir(rng, ACENTOS)
    serv = rng.choice([("Compañía de Energía del Plata", "kWh"), ("Aguas Corrientes del Sur", "m³"), ("Red de Gas Oriental", "m³"), ("Telecomunicaciones del Este", "GB")])
    n = lambda v: fmt_num(v, "uy")
    m = 45
    lz.d.rectangle([0, 0, W, 90], fill=acc)
    lz.t(m, 26, serv[0], 22, True, color=(255, 255, 255))
    y = 110
    lz.t(m, y, "Cliente:", 11, True)
    lz.t(m + 80, y, f"{P['nombres']} {P['apellidos']}".upper(), 11)
    lz.t(m, y + 18, "Dirección:", 11, True)
    lz.t(m + 80, y + 18, f"{elegir(rng, CALLES)} {rng.randint(100, 4000)} - {elegir(rng, CIUDADES)}", 11)
    lz.t(m, y + 36, "Nº de cuenta:", 11, True)
    lz.t(m + 100, y + 36, str(rng.randint(10 ** 7, 10 ** 9)), 11)
    lz.r(W * .6, y - 6, W - m, y + 80, 2, acc)
    total = rng.uniform(600, 9000)
    lz.t(W * .6 + 14, y + 2, "TOTAL A PAGAR", 12, True)
    lz.t(W - m - 14, y + 24, f"$ {n(total)}", 22, True, "r")
    lz.t(W * .6 + 14, y + 56, f"Vencimiento: {fecha(rng.randint(1, 28), rng.randint(1, 12), rng.randint(2019, 2026))}", 11)
    y += 110
    lz.t(m, y, "Historial de consumo", 12, True)
    y += 24
    vals = [rng.uniform(100, 600) for _ in range(12)]
    bw = (W - 2 * m) / 12
    for i, v in enumerate(vals):
        bh = v / 600 * 140
        lz.d.rectangle([m + i * bw + 6, y + 150 - bh, m + (i + 1) * bw - 6, y + 150], fill=acc)
        lz.t(m + i * bw + bw / 2, y + 156, MESES[i][:3], 9, False, "c")
    y += 190
    cols = [("Concepto", .6, "l"), ("Cantidad", .2, "r"), ("Importe", .2, "r")]
    consumo = rng.randint(80, 900)
    filas = [["Cargo fijo", "", n(total * .2)], [f"Consumo ({serv[1]})", str(consumo), n(total * .55)], ["Cargo por potencia / servicio", "", n(total * .07)], ["IVA 22%", "", n(total * .18)]]
    y = tabla(lz, m, y, W - 2 * m, cols, filas, 22, 11, "header", (240, 240, 240), color_borde=acc) + 30
    lz.parrafo(m, y, W - 2 * m, " ".join(rng.sample(FRASES, 3)), 10, color=(100, 100, 100))
    _barcode(lz, m, H - 120, W * .6, 60)
    return lz.img


def neg_planilla_horas(rng):
    """Planilla de control de horas: tiene datos del empleado pero no es un recibo (negativo difícil)."""
    W, H = 900, 1273
    lz = Lienzo(rng, W, H)
    E, P = empresa(rng), persona(rng)
    m = 45
    y = 40
    lz.t(m, y, E["nombre"], 15, True)
    lz.t(W - m, y, rng.choice(["PLANILLA DE CONTROL DE HORAS", "REGISTRO DE ASISTENCIA", "PARTE DIARIO DE HORAS"]), 14, True, "r")
    y += 40
    mes, anio = rng.randint(1, 12), rng.randint(2019, 2026)
    for k, v in [("Funcionario", f"{P['apellidos']}, {P['nombres']}"), ("C.I.", P["ci"]), ("Sector", P["sector"]), ("Mes", f"{MESES[mes - 1]} {anio}")]:
        lz.t(m, y, f"{k}:", 11, True)
        lz.t(m + 100, y, v, 11)
        y += 18
    y += 14
    cols = [("Día", .1, "c"), ("Entrada", .15, "c"), ("Salida almuerzo", .17, "c"), ("Regreso", .15, "c"), ("Salida", .15, "c"), ("Horas", .1, "c"), ("Firma", .18, "c")]
    filas = []
    tot = 0
    for d in range(1, rng.randint(22, 31) + 1):
        if rng.random() < .25:
            filas.append([str(d), "", "", "", "", "", ""])
            continue
        e = rng.choice([7, 8, 8, 9])
        h = rng.choice([8, 8, 9, 6])
        tot += h
        filas.append([str(d), f"{e:02d}:{rng.choice(['00', '05', '10', '55'])}", "12:00", "13:00", f"{e + h + 1:02d}:{rng.choice(['00', '02', '15'])}", str(h), ""])
    y = tabla(lz, m, y, W - 2 * m, cols, filas, 27 if len(filas) < 26 else 25, 10, "full", (230, 230, 230))
    y += 12
    lz.t(W - m, y, f"Total horas del mes: {tot}", 12, True, "r")
    lz.t(m, y + 40, "Firma del supervisor: ____________________", 11)
    return lz.img


def neg_recibo_pago(rng):
    """Recibo de pago genérico (alquiler, cuota, servicio): dice 'recibo' pero no es de sueldo (negativo difícil)."""
    W, H = 1000, rng.randint(560, 720)
    lz = Lienzo(rng, W, H)
    P = persona(rng)
    acc = elegir(rng, ACENTOS)
    monto = rng.uniform(2000, 60000)
    m = 40
    lz.r(15, 15, W - 15, H - 15, 2, acc)
    y = 40
    lz.t(m, y, "RECIBO", 30, True, color=acc)
    lz.t(m + 180, y + 12, f"Nº {rng.randint(1, 9999):04d}", 16)
    lz.r(W - 300, y, W - m, y + 50, 2, acc)
    lz.t(W - 290, y + 12, "$", 22, True)
    lz.t(W - m - 12, y + 12, fmt_num(monto, "uy"), 22, True, "r")
    y += 90
    concepto = rng.choice(["alquiler del inmueble", "cuota social", "honorarios profesionales", "servicio de transporte escolar",
                           "cuota del curso", "seña por compra de vehículo", "reparación de vivienda", "gastos comunes"])
    texto = (f"Recibí de {P['nombres']} {P['apellidos']} la cantidad de pesos uruguayos {en_letras(monto)} "
             f"por concepto de {concepto} correspondiente al mes de {elegir(rng, MESES).lower()} de {rng.randint(2019, 2026)}.")
    y = lz.parrafo(m, y, W - 2 * m, texto, 16, 2.0, rng.choice(["serif", "sans"]))
    y += 30
    lz.t(m, y, f"{elegir(rng, CIUDADES)}, {rng.randint(1, 28)} de {elegir(rng, MESES).lower()} de {rng.randint(2019, 2026)}", 13)
    firma(lz, W - 330, y - 10, 200, 50)
    lz.l(W - 360, y + 50, W - m, y + 50, 1)
    lz.t(W - 200, y + 56, "Firma y aclaración", 11, False, "c")
    return lz.img


def neg_carta(rng):
    W, H = 900, 1273
    lz = Lienzo(rng, W, H)
    E = empresa(rng)
    acc = elegir(rng, ACENTOS)
    m = 90
    y = 60
    if rng.random() < .7:
        logo(lz, m, y, 44, E["iniciales"], acc)
        lz.t(m + 56, y + 8, E["nombre"], 15, True, color=acc)
    y += 100
    lz.t(W - m, y, f"{E['ciudad']}, {rng.randint(1, 28)} de {elegir(rng, MESES).lower()} de {rng.randint(2019, 2026)}", 12, False, "r")
    y += 50
    lz.t(m, y, rng.choice(["Señores:", "Estimados clientes:", "A quien corresponda:", "MEMORÁNDUM INTERNO"]), 13, True)
    y += 20
    lz.t(m, y, nombre_empresa(rng) if rng.random() < .5 else "", 12)
    y += 40
    estilo = rng.choice(["serif", "sans"])
    pool = rng.sample(FRASES, len(FRASES))
    for _ in range(rng.randint(3, 6)):
        k = rng.randint(2, 4)
        if len(pool) < k:
            break
        y = lz.parrafo(m, y, W - 2 * m, " ".join(pool[:k]), 13, 1.6, estilo) + 16
        pool = pool[k:]
        if y > H - 260:
            break
    y += 20
    lz.t(m, y, "Saludamos a ustedes atentamente,", 13, estilo=estilo)
    firma(lz, m, y + 30, 180, 60)
    lz.t(m, y + 100, f"{elegir(rng, NOMBRES)} {elegir(rng, APELLIDOS)}", 12, True)
    lz.t(m, y + 118, rng.choice(["Gerencia General", "Departamento Comercial", "Administración"]), 11)
    return lz.img



# ---------------------------------------------------------------------------
# NEGATIVOS inspirados en la carpeta Doc_varios_no_recibos (todo ficticio)
# ---------------------------------------------------------------------------
COMERCIOS = ["SUPERMERCADO EL DORADO", "FARMACIA CENTRAL", "TIENDA LOS 33", "PANADERÍA RÚSTICA", "COMBUSTIBLES DEL ESTE",
             "RESTAURANTE EL PUERTO", "CINE CENTRO", "ROPA URBANA", "PEAJE RUTA 9", "STREAMING PLUS", "LIBRERÍA EL ÁNCORA",
             "GIMNASIO FORMA", "AUTOSERVICE LA ESQUINA", "FERRETERÍA EL CLAVO", "CARNICERÍA DON PEPE", "BAR LA PASIVA DEL SUR",
             "VERDULERÍA LA HUERTA", "PERFUMERÍA AROMAS", "SEGURO AUTOMOTOR", "TELEFONÍA MÓVIL", "COMPRA WEB MERCADO",
             "HELADERÍA POLO", "ÓPTICA VISIÓN", "DEPORTES GOL", "PIZZERÍA NAPOLI", "ELECTRO HOGAR CUOTA 3/12", "PROMO EN CUOTAS"]
MARCAS_TARJETA = ["NOVA", "CREDIX", "ORBE", "PLATA CARD", "UNICARD", "ALFA"]
DEPARTAMENTOS_FICT = ["Villa Esperanza", "Puertos Altos", "Cerro Bajo", "San Telmo del Sur", "Laguna Clara", "Arroyo Verde"]


def _marca_tarjeta(lz, x, y, nombre, size, color):
    """Marca de tarjeta inventada: wordmark en itálica negrita."""
    lz.t(x, y, nombre, size, True, color=color, italic=True)


def _stub_tarjeta(rng, w, h, banco, marca, acc, cuenta, cierre, venc, moneda):
    """Talón desprendible (se dibuja horizontal y después se rota 90°)."""
    st = Lienzo(rng, h, w, papel=(255, 255, 255))
    st.fams = {k: v for k, v in st.fams.items()}
    st.r(2, 2, h - 3, w - 3, 1, (120, 120, 120))
    st.t(10, 8, banco, 11, True, color=acc)
    _marca_tarjeta(st, h - 90, 6, marca, 13, (20, 40, 120))
    campos = [("Nº Cuenta", cuenta), ("Moneda", moneda), ("F. Cierre", cierre), ("F. Vencimiento", venc)]
    for i, (k, v) in enumerate(campos):
        x = 10 + i * (h - 20) / 4
        st.t(x, 30, k, 8, color=(90, 90, 90))
        st.t(x, 44, v, 11, True)
    _barcode(st, 10, w - 34, h * .7, 24)
    return st.img.rotate(90, expand=True)


def neg_tarjeta(rng):
    """Estado de cuenta de tarjeta de crédito, con talones desprendibles al costado (tipo Santander/Scotia)."""
    W, H = 1000, 1320
    lz = Lienzo(rng, W, H, papel=elegir(rng, [(255, 255, 255), (250, 250, 250), (252, 252, 248)]))
    P = persona(rng)
    banco = elegir(rng, BANCOS)
    marca = elegir(rng, MARCAS_TARJETA)
    acc = elegir(rng, [(200, 30, 40), (30, 60, 150), (0, 110, 90), (200, 90, 0)])
    fm = rng.choice(["uy", "uy", "en"])
    n = lambda v: fmt_num(v, fm)
    anio, mes = rng.randint(2019, 2026), rng.randint(1, 11)
    cierre = fecha(rng.randint(1, 28), mes, anio)
    venc = fecha(rng.randint(1, 28), mes + 1, anio)
    cuenta = str(rng.randint(10 ** 7, 10 ** 10))
    con_stubs = rng.random() < .8
    xr = W * .7 if con_stubs else W - 30
    m = 30
    y = 30
    _marca_tarjeta(lz, m, y, marca, 26, (20, 40, 120))
    logo(lz, m + 170, y, 34, "".join(w[0] for w in banco.split()[:2]), acc)
    lz.t(m + 212, y + 6, banco, 15, True, color=acc)
    lz.t(xr - 10, y, "SU ESTADO DE CUENTA", 13, True, "r")
    lz.t(xr - 10, y + 20, f"R.U.C. {rng.randint(10 ** 11, 10 ** 12 - 1)}", 9, False, "r")
    y += 64
    lz.t(m, y, f"{banco} S.A.", 9)
    lz.t(m, y + 12, f"{elegir(rng, CALLES)} {rng.randint(100, 2000)}", 9)
    lz.t(m, y + 24, "MONTEVIDEO - CP 11.100", 9)
    lz.t(xr - 10, y + 8, "Hoja Nº 1/1", 10, False, "r")
    y += 50
    lz.t(m, y, f"{P['apellidos']} {P['nombres']}".upper(), 11)
    lz.t(m, y + 15, f"{elegir(rng, CALLES)} {rng.randint(100, 4000)}".upper(), 11)
    lz.t(m, y + 30, f"{rng.randint(11000, 11900)} - {elegir(rng, CIUDADES).upper()} - URUGUAY", 11)
    _barcode(lz, m, y + 50, 150, 16)
    for i, (k, v) in enumerate([("Fecha Emisión", cierre), ("Fecha Cierre", cierre), ("Vencimiento Actual", venc)]):
        yy = y + i * 34
        lz.r(xr - 170, yy - 4, xr - 10, yy + 28, 1, (130, 130, 130))
        lz.t(xr - 164, yy - 2, k, 8, color=(90, 90, 90))
        lz.t(xr - 90, yy + 10, v, 12, True, "c")
    y += 110
    lim = rng.choice([30000, 50000, 80000, 100000, 150000, 250000])
    grid = [("Nº Cuenta", cuenta), ("Límite de Crédito", n(lim)), ("Crédito disponible", n(lim * rng.uniform(.2, 1))),
            ("Próximo Cierre", fecha(rng.randint(1, 28), mes + 1, anio)), ("Próximo Vencimiento", fecha(rng.randint(1, 28), min(12, mes + 2), anio))]
    cw = (xr - 10 - m) / len(grid)
    lz.r(m, y, xr - 10, y + 40, 1, (130, 130, 130))
    for i, (k, v) in enumerate(grid):
        x = m + i * cw
        if i:
            lz.l(x, y, x, y + 40, 1, (130, 130, 130))
        lz.t(x + 6, y + 3, k, 8, color=(90, 90, 90))
        lz.t(x + cw / 2, y + 18, v, 12, True, "c")
    y += 52
    saldo_ant = rng.uniform(0, lim * .6)
    movs = []
    for _ in range(rng.randint(2, 26)):
        d = fecha(rng.randint(1, 28), mes, anio)[:5]
        usd = rng.random() < .15
        mto = rng.uniform(80, 9000) if not usd else rng.uniform(3, 200)
        movs.append((d, rng.choice(["", f"{rng.randint(0, 9999):04d}"]), elegir(rng, COMERCIOS), 0.0 if usd else mto, mto if usd else 0.0))
    movs.sort()
    cargos = sum(x[3] for x in movs)
    pago_cont = saldo_ant + cargos
    bloques = [("Saldo Anterior", saldo_ant), ("Cargos del mes", cargos), ("Pago Total Contado", pago_cont), ("Pago Mínimo", pago_cont * rng.uniform(.05, .15))]
    bw = (xr - 10 - m) / 4
    for i, (k, v) in enumerate(bloques):
        x = m + i * bw
        lz.r(x, y, x + bw - 6, y + 46, 1, (130, 130, 130))
        lz.t(x + 6, y + 3, k, 8, color=(90, 90, 90))
        lz.t(x + 6, y + 18, "$", 9)
        lz.t(x + bw - 14, y + 16, n(v), 11, True, "r")
        lz.t(x + 6, y + 31, "U$S", 9)
        lz.t(x + bw - 14, y + 30, n(0), 10, False, "r")
    y += 58
    tea = rng.uniform(40, 75)
    lz.r(m, y, xr - 10, y + 36, 1, (130, 130, 130))
    lz.t(m + 6, y + 3, "Tasa Efectiva Anual actual", 8, color=(90, 90, 90))
    lz.t(m + 6, y + 18, f"Financiación {tea:.1f}%   Mora {tea * 1.2:.1f}%   U$S {tea / 5:.1f}%".replace(".", ","), 10)
    lz.t(xr - 200, y + 3, "Cuotas a Vencer", 8, color=(90, 90, 90))
    lz.t(xr - 20, y + 18, n(rng.choice([0, rng.uniform(500, 20000)])), 10, False, "r")
    y += 50
    cols = [("Fecha", .1, "l"), ("Tarjeta", .1, "l"), ("Detalle", .44, "l"), ("Importe Pesos", .18, "r"), ("Importe Dólares", .18, "r")]
    filas = [["", "", "SALDO ANTERIOR", n(saldo_ant), n(0)]] + [[a, b, c, n(d) if d else "", n(e) if e else ""] for a, b, c, d, e in movs]
    if rng.random() < .5:
        filas.append(["", "", "PAGOS", n(-saldo_ant), ""])
    tabla(lz, m, y, xr - 10 - m, cols, filas, 15, 9, rng.choice(["header", "none"]), None, estilo=rng.choice(["mono", "sans"]))
    lz.parrafo(m, H - 60, xr - 10 - m, "Abonar únicamente los pagos mínimos aumentará significativamente el tiempo en que llevará pagar su deuda así como el monto de la misma. Este comprobante es emitido a título informativo.", 7, color=(110, 110, 110))
    if con_stubs:
        lz.l(xr, 20, xr, H - 20, 1, (160, 160, 160))
        sh = (H - 60) // 4
        for i, mon in enumerate(["Pesos", "Dólares", "Pesos", "Dólares"]):
            st = _stub_tarjeta(rng, int(W - xr - 20), sh - 10, banco, marca, acc, cuenta, cierre, venc, mon)
            lz.img.paste(st, (int(xr + 10), int(20 + i * sh)))
        lz.d = ImageDraw.Draw(lz.img)
    return lz.img


def neg_tributo(rng):
    """Factura de tributo municipal (contribución inmobiliaria / tributos domiciliarios / patente)."""
    W, H = 900, 1300
    lz = Lienzo(rng, W, H)
    P = persona(rng)
    depto = elegir(rng, DEPARTAMENTOS_FICT)
    tipo = rng.choice(["TRIBUTOS DOMICILIARIOS", "CONTRIBUCIÓN INMOBILIARIA", "PATENTE DE RODADOS", "TASA GENERAL MUNICIPAL"])
    acc = elegir(rng, [(40, 40, 40), (30, 60, 120), (20, 80, 60)])
    n = lambda v: fmt_num(v, "uy")
    m = 50
    y = 40
    logo(lz, m, y, 40, "ID", acc)
    lz.t(m + 50, y + 2, "Intendencia", 13, True, color=acc)
    lz.t(m + 50, y + 18, depto, 13, color=acc)
    lz.t(W - m, y, tipo, 12, True, "r")
    _barcode(lz, W - m - 230, y + 20, 230, 28)
    lz.t(W - m, y + 52, f"CM{rng.randint(10 ** 9, 10 ** 10)}UY", 8, False, "r")
    y += 80
    es = "mono"
    anio, mes = rng.randint(2019, 2026), rng.randint(1, 11)
    lz.t(m + 10, y, f"C.P.: {rng.randint(11000, 11900)} > {rng.randint(10000, 99999)} <       {anio}{mes:02d}{rng.randint(10, 28)} 00-{rng.randint(10000, 99999):05d}-{rng.randint(100, 999)}-{rng.randint(10 ** 6, 10 ** 7)}", 11, estilo=es)
    y += 30
    lz.t(m + 10, y, f"{elegir(rng, CALLES).upper()} {rng.randint(100, 4000)} {rng.choice(['', 'AP.' + str(rng.randint(1, 12))])}", 11, True, estilo=es)
    y += 30
    if "PATENTE" in tipo:
        lz.t(m + 10, y, f"MATRÍCULA {rng.choice('ABCDEFGHIJKLMNOPQRS')}{rng.choice('ABCDEFGHIJKLMNOPQRS')}{rng.choice('ABCDEFGHIJKLMNOPQRS')} {rng.randint(1000, 9999)}   PADRÓN {rng.randint(100000, 999999)}", 10, estilo=es)
    else:
        lz.t(m + 10, y, f"CARPETA CATASTRAL {rng.randint(1000, 9999)} PADRÓN {rng.randint(10000, 99999)} SOLAR {rng.randint(1, 60)}", 10, estilo=es)
    lz.t(m + 10, y + 16, f"{P['apellidos']} {P['nombres']}".upper(), 10, estilo=es)
    y += 50
    cab = [("REFERENCIA DE COBRO", f"M{rng.randint(10 ** 12, 10 ** 13)}"), ("NRO. DE DOCUMENTO", f"02-{anio}{mes:02d}0-{rng.randint(10 ** 6, 10 ** 7)}"),
           ("FECHA DE CIERRE", fecha(rng.randint(1, 28), mes, anio)), ("NDP", str(rng.randint(10 ** 7, 10 ** 8))), ("CUENTA CORRIENTE", str(rng.randint(10 ** 5, 10 ** 7)))]
    cw = (W - 2 * m) / 5
    lz.r(m, y, W - m, y + 50, 1)
    lz.d.rectangle([m + 1, y + 1, W - m - 1, y + 20], fill=(225, 225, 225))
    for i, (k, v) in enumerate(cab):
        x = m + i * cw
        if i:
            lz.l(x, y, x, y + 50, 1)
        lz.t(x + cw / 2, y + 5, k, 8, False, "c")
        lz.t(x + cw / 2, y + 28, v, 10, True, "c", es)
    y += 64
    alto = rng.randint(420, 560)
    x1, x2, x3 = m + (W - 2 * m) * .34, m + (W - 2 * m) * .66, m + (W - 2 * m) * .83
    lz.r(m, y, W - m, y + alto, 1)
    lz.d.rectangle([m + 1, y + 1, W - m - 1, y + 22], fill=(225, 225, 225))
    for xx in (x1, x2, x3):
        lz.l(xx, y, xx, y + alto, 1)
    lz.t((m + x1) / 2, y + 5, "INFORMACIÓN", 9, False, "c")
    lz.t((x1 + x2) / 2, y + 5, "CONCEPTO", 9, False, "c")
    lz.t((x2 + x3) / 2, y + 5, "DEUDA ANTERIOR", 9, False, "c")
    lz.t((x3 + W - m) / 2, y + 5, "DEUDA ACTUAL", 9, False, "c")
    conc = {"TRIBUTOS DOMICILIARIOS": ["TASA GENERAL", "ADIC. ILUMINACIÓN", "IMPRES. Y ENVÍO FACT", "REDONDEO"],
            "CONTRIBUCIÓN INMOBILIARIA": ["CONTRIBUCIÓN INMOB.", "ADICIONAL MERCANTIL", "TASA SANEAMIENTO", "RECARGOS", "REDONDEO"],
            "PATENTE DE RODADOS": ["PATENTE CUOTA " + str(rng.randint(1, 6)), "MULTAS", "RECARGO MORA", "REDONDEO"],
            "TASA GENERAL MUNICIPAL": ["TASA GENERAL", "TASA BROMATOLÓGICA", "IMPRES. Y ENVÍO FACT", "REDONDEO"]}[tipo]
    tot = 0
    yy = y + 30
    for c in conc:
        v = rng.uniform(-0.9, 0.9) if c == "REDONDEO" else rng.uniform(30, 9000)
        tot += v
        lz.t(x1 + 6, yy, c, 8, estilo=es)
        lz.t(W - m - 8, yy, n(v), 8, False, "r", es)
        if rng.random() < .2 and c != "REDONDEO":
            lz.t(x3 - 8, yy, n(v * rng.uniform(.5, 1.5)), 8, False, "r", es)
        yy += 13
    lz.t(m + 10, y + alto - 40, f"www.{depto.lower().replace(' ', '')}.gub.uy", 9, False)
    lz.t(m + 10, y + alto - 26, "Fonotributo: 1950 3000", 8)
    y += alto
    lz.r(x2, y, W - m, y + 46, 1)
    lz.l(x3 - 30, y, x3 - 30, y + 46, 1)
    lz.t(x2 + 6, y + 4, "FECHA VENCIMIENTO", 8)
    lz.t(x3 - 24, y + 4, "TOTAL A PAGAR", 8)
    lz.t(x2 + 6, y + 22, fecha(rng.randint(1, 28), mes + 1, anio), 12, True, estilo=es)
    lz.t(W - m - 8, y + 22, "*" * rng.randint(4, 9) + n(tot), 12, True, "r", es)
    y += 56
    lz.parrafo(m, y, W * .45, "INFORMACIÓN IMPORTANTE AL DORSO. Este documento adquiere valor de recibo únicamente con la autenticación de cobro.", 8, estilo=es)
    lz.parrafo(W * .52, y, W * .42, "El pago de este documento con posterioridad a la Fecha de Vencimiento implicará la aplicación de Multas y Recargos.", 8, estilo=es)
    y += 50
    lz.punteada(m, y, W - m, 4, (100, 100, 100))
    y += 10
    lz.t(W / 2, y, "TALÓN PARA EL CENTRO DE COBRO", 9, False, "c", es)
    y += 20
    for k, v in [("LÍMITE DE RECEPCIÓN", fecha(rng.randint(1, 28), mes + 1, anio)), ("REFERENCIA COBRO", cab[0][1]), ("TOTAL A PAGAR", "******" + n(tot)), ("CÓDIGO DE COBRANZA", str(rng.randint(10 ** 5, 10 ** 6)))]:
        lz.t(m + 10, y, k, 8, estilo=es)
        lz.t(m + 10, y + 12, v, 11, True, estilo=es)
        y += 32
    _barcode(lz, W * .35, y - 120, W * .5, 40)
    _barcode(lz, W * .35, y - 60, W * .5, 40)
    return lz.img


def neg_factura_agua(rng):
    """Factura de servicio (agua/energía/gas) tipo e-Ticket con detalle, adenda con gráfico y talón de cobranza."""
    W, H = 900, 1300
    lz = Lienzo(rng, W, H)
    P = persona(rng)
    serv = rng.choice([("Aguas Corrientes del Sur", "ACS", "m³", "Agua Potable"), ("Compañía de Energía del Plata", "CEP", "kWh", "Energía"),
                       ("Red de Gas Oriental", "RGO", "m³", "Gas Natural"), ("Servicio Sanitario del Litoral", "SSL", "m³", "Agua Potable")])
    acc = elegir(rng, [(30, 90, 170), (0, 120, 160), (40, 70, 140), (200, 100, 0)])
    n = lambda v: fmt_num(v, "uy")
    m = 40
    y = 30
    for i, tx in enumerate([f"Administración de {serv[0]}", f"Montevideo: {elegir(rng, CALLES)} {rng.randint(100, 2000)} - Facturación", "Interior del País: Oficinas Locales"]):
        lz.t(m, y + i * 13, tx, 9, i == 0, color=acc)
    lz.d.rectangle([W * .45, y, W * .45 + 50, y + 40], fill=acc)
    lz.t(W * .45 + 25, y + 10, serv[1], 14, True, "c", color=(255, 255, 255))
    lz.t(W * .45 + 25, y + 44, serv[0], 10, True, "c", color=acc)
    lz.t(W - m, y, f"info@{serv[1].lower()}.com.uy", 9, False, "r", color=acc)
    lz.t(W - m, y + 14, f"0800 {rng.randint(1000, 9999)}  Líneas gratuitas", 9, False, "r", color=acc)
    y += 72
    _barcode(lz, m, y, 130, 22)
    lz.t(m, y + 26, f"Ref. Cobro: {rng.randint(10 ** 8, 10 ** 9)}", 7)
    lz.t(m + 150, y, f"{P['apellidos']}, {P['nombres']}".upper(), 10, True)
    dire = f"{elegir(rng, CALLES)} {rng.randint(100, 4000)}".upper()
    lz.t(m + 150, y + 14, dire, 9)
    lz.t(m + 150, y + 27, f"{rng.randint(11000, 11900)}, {elegir(rng, CIUDADES).upper()}", 9)
    lz.t(W - m, y - 4, f"R.U.T. 21{rng.randint(10 ** 9, 10 ** 10 - 1)}", 13, True, "r", color=acc)
    lz.t(W - m - 150, y + 16, "E-TICKET", 11)
    lz.t(W - m - 150, y + 30, f"{rng.choice('ABQZ')} {rng.randint(10 ** 6, 10 ** 7)} CRÉDITO", 11)
    anio, mes = rng.randint(2019, 2026), rng.randint(1, 11)
    vence = fecha(rng.randint(1, 28), mes + 1, anio)
    lz.t(W - m - 150, y + 44, f"VENCE: {vence}", 11)
    y += 70
    lz.r(m, y, m + 200, y + 46, 1)
    lz.t(m + 8, y + 4, "CONSUMIDOR FINAL", 12, True)
    lz.t(m + 30, y + 24, str(rng.randint(10 ** 7, 10 ** 8)), 12)
    lz.r(m + 200, y, W - m - 160, y + 22, 1)
    lz.t(m + 206, y + 3, "TITULAR", 9, color=acc)
    lz.t(m + 206, y + 12, f"{P['apellidos']}, {P['nombres']}".upper(), 8)
    lz.r(W - m - 160, y, W - m, y + 22, 1)
    lz.t(W - m - 154, y + 4, "TIPO DOCUMENTO  PAÍS", 9, color=acc)
    y += 52
    for i, (k, v) in enumerate([("TARIFA", "Residencial"), ("UNIDADES", "1"), ("DIÁMETRO" if serv[2] == "m³" else "POTENCIA", "12.5 mm" if serv[2] == "m³" else f"{rng.choice([3.3, 4.4, 6.6, 9.2])} kW")]):
        lz.t(m + 8, y + i * 12, f"{k}: {v}", 8)
    hdr = [("Nº DE CUENTA", str(rng.randint(10 ** 7, 10 ** 8))), ("FECHA", fecha(rng.randint(1, 28), mes, anio)),
           ("PERÍODO CONSUMO", f"{rng.randint(1, 28):02d}/{mes:02d}/{anio}"), ("VENCIMIENTO", vence)]
    cw = (W - m - (m + 220)) / 4
    for i, (k, v) in enumerate(hdr):
        x = m + 220 + i * cw
        lz.d.rectangle([x, y, x + cw - 4, y + 14], fill=acc)
        lz.t(x + cw / 2, y + 1, k, 8, True, "c", color=(255, 255, 255))
        lz.r(x, y + 14, x + cw - 4, y + 34, 1, acc)
        lz.t(x + cw / 2, y + 17, v, 11, True, "c")
    y += 48
    xp = W - m - 140
    alto = rng.randint(300, 420)
    lz.r(m, y, W - m, y + alto, 1, acc)
    lz.l(xp, y, xp, y + alto, 1, acc)
    lz.d.rectangle([m + 1, y + 1, xp - 1, y + 16], fill=(230, 236, 245))
    lz.t((m + xp) / 2, y + 2, "DETALLE DE LA FACTURACIÓN", 10, True, "c", color=acc)
    lz.d.rectangle([xp + 1, y + 1, W - m - 1, y + 16], fill=acc)
    lz.t((xp + W - m) / 2, y + 2, "PRECIO FINAL", 10, True, "c", color=(255, 255, 255))
    consumo = rng.randint(5, 30) if serv[2] == "m³" else rng.randint(90, 900)
    items = [(f"Cargo Fijo {serv[3]}", rng.uniform(150, 600)), (f"Consumo {serv[3]} ({consumo} {serv[2]})", consumo * rng.uniform(15, 40) if serv[2] == "m³" else consumo * rng.uniform(5, 9)),
             ("Costo de impresión y envío", rng.uniform(20, 60)), ("Ajuste por Redondeo", rng.uniform(-.9, .9))]
    yy = y + 24
    tot = 0
    for c, v in items:
        lz.t(m + 10, yy, c, 9)
        lz.t(m + 24, yy + 11, f"{rng.randint(1, 28):02d}/{mes:02d}/{anio} a {rng.randint(1, 28):02d}/{mes + 1:02d}/{anio}: 1,00 x {n(abs(v))}", 7, color=(90, 90, 90))
        lz.t(W - m - 8, yy, n(v), 9, False, "r")
        tot += v
        yy += 26
    iva = tot * .22
    yb = y + alto - 80
    for k, v in [("Importe No Gravado", tot * .1), ("Importe Gravado", tot * .9), ("IVA Tasa Básica", iva), ("Total Monto", tot + iva)]:
        lz.t(m + 10, yb, k, 9)
        lz.t(W - m - 8, yb, n(v), 9, False, "r")
        yb += 15
    y += alto + 8
    lz.r(m, y, xp - 10, y + 30, 1, acc)
    lz.t(m + 8, y + 4, "REFERENCIAS:", 9, color=acc)
    lz.d.rectangle([xp, y, W - m, y + 14], fill=acc)
    lz.t((xp + W - m) / 2, y + 1, "TOTAL", 10, True, "c", color=(255, 255, 255))
    lz.r(xp, y + 14, W - m, y + 40, 1, acc)
    lz.t((xp + W - m) / 2, y + 20, f"$***{n(tot + iva)}", 11, True, "c")
    y += 54
    _qr(lz, m, y, 80)
    lz.t(m + 95, y + 6, "Puede verificar comprobante en el sitio web", 8)
    lz.t(m + 95, y + 18, f"Número de CAE: {rng.randint(10 ** 10, 10 ** 11)}", 8)
    lz.t(m + 95, y + 50, f"Código de seguridad: {''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjk0123456789') for _ in range(6))}", 12)
    lz.r(W * .55, y + 40, W * .8, y + 72, 1)
    lz.t(W * .675, y + 44, "Fecha de vencimiento", 10, False, "c")
    lz.t(W * .675, y + 57, f"CAE: {fecha(rng.randint(1, 28), rng.randint(1, 12), anio + 2)}", 10, False, "c")
    y += 92
    lz.r(m, y, W - m, y + 110, 1, (150, 150, 150))
    lz.t(m + 8, y + 4, "ADENDA", 10, True, color=acc)
    for i, (k, v) in enumerate([("NUM. MEDIDOR", f"TGA{rng.randint(100000, 999999)}"), ("LEC. ANTERIOR", str(rng.randint(100, 900))), ("LEC. ACTUAL", str(rng.randint(100, 900))), (f"CONSUMO {serv[2]}", str(consumo))]):
        lz.t(m + 10, y + 24 + i * 13, k, 8)
        lz.t(m + 110, y + 24 + i * 13, v, 8)
    gx, gy = m + 260, y + 90
    for i in range(8):
        for j, col in enumerate([(60, 60, 60), (150, 150, 150)]):
            h = rng.uniform(10, 60)
            lz.d.rectangle([gx + i * 26 + j * 9, gy - h, gx + i * 26 + j * 9 + 7, gy], fill=col)
    lz.l(gx - 4, gy, gx + 210, gy, 1)
    lz.parrafo(W * .68, y + 20, W * .27, "Esta representación impresa adquiere valor de recibo sólo si posee autenticación de pago.", 7)
    y += 122
    lz.punteada(m, y, W - m, 4, (100, 100, 100))
    lz.t(m, y + 6, "Talón de cobranza. No doble ni deteriore este talón para evitar una lectura errónea.", 8, True, color=acc)
    y += 24
    for k, x in [("PAGO DEL MES", m), ("PAGO TOTAL", W / 2 + 10)]:
        lz.d.rectangle([x, y, x + 46, y + 30], fill=acc)
        lz.t(x + 23, y + 8, serv[1], 10, True, "c", color=(255, 255, 255))
        lz.t(x, y + 36, k, 10, color=acc)
        for j, (kk, vv) in enumerate([("Nº DE CUENTA", hdr[0][1] if x == m else "XXXXXXX"), ("IMPORTE", f"$***{n(tot + iva)}" if x == m else "XXXXXXX")]):
            xx = x + 70 + j * 140
            lz.d.rectangle([xx, y, xx + 120, y + 12], fill=acc)
            lz.t(xx + 60, y, kk, 8, True, "c", color=(255, 255, 255))
            lz.t(xx + 60, y + 16, vv, 9, False, "c")
    _barcode(lz, m, y + 60, W * .55, 34)
    lz.t(m, y + 98, "".join(str(rng.randint(0, 9)) for _ in range(40)), 9, estilo="mono")
    return lz.img


def plantilla_vacia(rng):
    """Plantilla genérica de recibo de haberes SIN datos."""
    apaisada = rng.random() < .5  # media hoja apaisada, como muchos recibos impresos
    W, H = (1100, rng.randint(720, 820)) if apaisada else (900, rng.randint(1000, 1273))
    lz = Lienzo(rng, W, H)
    m = 50
    y = 30 if apaisada else 40
    lz.t(W / 2, y, rng.choice(["RECIBO DE HABERES", "RECIBO DE SUELDO", "LIQUIDACIÓN DE HABERES"]), 18, True, "c")
    y += 40
    campos = [("Empresa:", "RUT:"), ("Domicilio:", "B.P.S.:"), ("Funcionario:", "C.I.:"), ("Cargo:", "Fecha de ingreso:"), ("Período:", "Fecha de pago:")]
    for a, b in campos:
        lz.t(m, y, a, 12)
        lz.l(m + lz.ancho(a, 12) + 6, y + 14, W / 2 - 20, y + 14, 1, (140, 140, 140))
        lz.t(W / 2, y, b, 12)
        lz.l(W / 2 + lz.ancho(b, 12) + 6, y + 14, W - m, y + 14, 1, (140, 140, 140))
        y += 26
    y += 16
    cols = [("Código", .1, "l"), ("Concepto", .42, "l"), ("Cantidad", .12, "r"), ("Haberes", .18, "r"), ("Descuentos", .18, "r")]
    filas = [[""] * 5 for _ in range(rng.randint(5, 8) if apaisada else rng.randint(10, 18))]
    y = tabla(lz, m, y, W - 2 * m, cols, filas, 22, 11, rng.choice(["full", "vert"]), rng.choice([None, (230, 230, 230)]),
              zebra=(245, 245, 245) if rng.random() < .4 else None)
    y += 12
    for k in ["Total Haberes", "Total Descuentos", "LÍQUIDO A COBRAR"]:
        lz.t(W - m - 200, y + 4, k, 12, k.isupper(), "r")
        lz.r(W - m - 190, y, W - m, y + 24, 1)
        lz.t(W - m - 184, y + 4, "$", 12)
        y += 30
    y += 30
    lz.parrafo(m, y, W * .5, "Recibí el importe líquido de la presente liquidación y copia de la misma, no teniendo nada que reclamar.", 10)
    lz.t(W - m - 120, y + 50, "Firma", 10, False, "c")
    lz.l(W - m - 240, y + 46, W - m, y + 46, 1)
    return lz.img


def neg_recibo_vacio(rng):
    """Recibo de sueldo SIN completar: parece un recibo pero no tiene validez (negativo difícil)."""
    r = rng.random()
    if r < .5:
        return fam_formulario(rng, completo=False)
    return plantilla_vacia(rng)


# CATEGORÍAS DE NEGATIVOS: cada una puede tener varias variantes (subtipos).
# Los documentos de Doc_varios_no_recibos se usaron como ejemplo para enriquecer
# la categoría que les corresponde (OSE -> factura, Santander/Scotia -> estado_cuenta,
# Intendencia -> tributo_municipal, recibos vacíos -> recibo_vacio).
NEGATIVOS = {
    "factura": [("e_factura", neg_factura), ("factura_servicio_eticket", neg_factura_agua)],
    "estado_cuenta": [("cuenta_bancaria", neg_estado_cuenta), ("tarjeta_credito", neg_tarjeta)],
    "ticket": [("ticket", neg_ticket)],
    "presupuesto": [("presupuesto", neg_presupuesto)],
    "constancia_laboral": [("constancia_laboral", neg_constancia)],
    "factura_servicios": [("factura_servicios", neg_factura_servicios)],
    "planilla_horas": [("planilla_horas", neg_planilla_horas)],
    "recibo_pago": [("recibo_pago", neg_recibo_pago)],
    "carta": [("carta", neg_carta)],
    "tributo_municipal": [("tributo_municipal", neg_tributo)],
    "recibo_vacio": [("formulario_vacio", lambda rng: fam_formulario(rng, completo=False)), ("plantilla_vacia", plantilla_vacia)],
}

# ---------------------------------------------------------------------------
# AUGMENTATIONS (idénticas para positivos y negativos)
# ---------------------------------------------------------------------------


def _coefs_perspectiva(dst, src):
    A, B = [], []
    for (x, y), (X, Y) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
        B += [X, Y]
    return np.linalg.solve(np.array(A, float), np.array(B, float)).tolist()


def _fondo(rng, npr, w, h):
    tipo = rng.choice(["madera", "liso", "mantel", "escritorio", "oscuro"])
    if tipo == "madera":
        base = np.array([rng.randint(110, 170), rng.randint(70, 110), rng.randint(40, 70)], float)
        yy = np.arange(h)[:, None]
        fase = npr.normal(0, 1, (h, 1)).cumsum(0) * .08 + np.linspace(0, rng.uniform(2, 8), w)[None, :]
        vetas = np.sin(yy / rng.uniform(6, 20) + fase) * 9
        arr = base[None, None, :] + vetas[..., None] + npr.normal(0, 6, (h, w, 1))
    elif tipo == "mantel":
        c1 = np.array([rng.randint(150, 230)] * 3, float)
        arr = np.ones((h, w, 3)) * c1
        paso = rng.randint(15, 40)
        arr[(np.arange(h) % paso) < 2, :, :] -= 50
        arr[:, (np.arange(w) % paso) < 2, :] -= 50
        arr += npr.normal(0, 6, (h, w, 1))
    elif tipo == "oscuro":
        arr = np.ones((h, w, 3)) * rng.randint(20, 70) + npr.normal(0, 6, (h, w, 1))
    else:
        c = np.array([rng.randint(150, 235), rng.randint(150, 235), rng.randint(150, 235)], float)
        arr = np.ones((h, w, 3)) * c + npr.normal(0, 5, (h, w, 1))
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img.filter(ImageFilter.GaussianBlur(rng.uniform(.5, 2)))


def _iluminacion(rng, npr, img, fuerza=(.65, .95)):
    w, h = img.size
    arr = np.asarray(img).astype(float)
    th = rng.uniform(0, 2 * math.pi)
    xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    g = xx * math.cos(th) + yy * math.sin(th)
    lo, hi = rng.uniform(*fuerza), 1.0
    gain = lo + (hi - lo) * (g - g.min()) / (np.ptp(g) + 1e-6)
    if rng.random() < .45:  # sombra suave (mano/celular)
        cx, cy = rng.uniform(-.5, 1.2), rng.uniform(-.2, 1.2)
        r = rng.uniform(.25, .6)
        d = np.sqrt((xx / 2 + .5 - cx) ** 2 + (yy / 2 + .5 - cy) ** 2)
        sombra = (d < r).astype(np.uint8) * 255
        sombra = np.asarray(Image.fromarray(sombra).filter(ImageFilter.GaussianBlur(max(w, h) * .08))) / 255
        gain = gain * (1 - rng.uniform(.08, .22) * sombra)
    tinte = np.array([rng.uniform(.97, 1.04), rng.uniform(.98, 1.02), rng.uniform(.93, 1.02)])
    arr = arr * gain[..., None] * tinte
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _ruido_jpeg(rng, npr, img, sigma=(0, 6), q=(45, 92)):
    arr = np.asarray(img).astype(float)
    s = rng.uniform(*sigma)
    if s > 0:
        arr = arr + npr.normal(0, s, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rng.randint(*q))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _arrugar(rng, npr, doc):
    """Arrugas y pliegues: relieve suave del papel -> leve deformación + sombreado."""
    w, h = doc.size
    # relieve de baja frecuencia (en float, sin cuantizar)
    celda = rng.randint(60, 160)
    chico = npr.normal(0, 1, (max(3, h // celda), max(3, w // celda))).astype(np.float32)
    z = np.asarray(Image.fromarray(chico, mode="F").resize((w, h), Image.BICUBIC), dtype=np.float32)
    try:
        import cv2
        z = cv2.GaussianBlur(z, (0, 0), celda / 3)
    except Exception:
        cv2 = None
    gy, gx = np.gradient(z)
    th = rng.uniform(0, 2 * math.pi)
    luz = gx * math.cos(th) + gy * math.sin(th)
    luz = luz / (luz.std() + 1e-6)
    sombreado = 1 + luz * rng.uniform(.03, .08)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    for _ in range(rng.randint(0, 2)):  # pliegues: línea fina con un lado apenas más oscuro
        ancho = rng.uniform(1.5, 4)
        if rng.random() < .5:
            d = yy - rng.uniform(.3, .7) * h
        else:
            d = xx - rng.uniform(.3, .7) * w
        sombreado += -rng.uniform(.06, .15) * np.exp(-(d / ancho) ** 2) + rng.choice([-1, 1]) * rng.uniform(.01, .04) * np.tanh(d / 30)
    arr = np.asarray(doc).astype(np.float32)
    if cv2 is not None:
        amp = rng.uniform(1, 3.5)
        gxn, gyn = gx / (np.abs(gx).max() + 1e-6), gy / (np.abs(gy).max() + 1e-6)
        arr = cv2.remap(arr, xx + gxn * amp, yy + gyn * amp, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    arr = arr * np.clip(sombreado, .75, 1.1)[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def aug_foto(rng, npr, doc):
    """Simula una foto de celular: documento sobre una superficie, en perspectiva, con luz despareja."""
    if rng.random() < .55:
        doc = _arrugar(rng, npr, doc)
    if rng.random() < .1:  # documento fotografiado de costado
        doc = doc.rotate(rng.choice([90, 270]), expand=True)
    dw, dh = doc.size
    # orientación del cuadro independiente del documento (evita que "apaisado" delate la clase)
    FW, FH = (1200, 900) if rng.random() < .35 else (900, 1200)
    fondo = _fondo(rng, npr, FW, FH)
    escala = (rng.uniform(.85, 1.02) if rng.random() < .1 else rng.uniform(.58, .82)) * min(FW / dw, FH / dh)
    sw, sh = dw * escala, dh * escala
    cx, cy = FW / 2 + rng.uniform(-.08, .08) * FW, FH / 2 + rng.uniform(-.08, .08) * FH
    ang = math.radians(rng.uniform(-12, 12))
    esquinas = [(-sw / 2, -sh / 2), (sw / 2, -sh / 2), (sw / 2, sh / 2), (-sw / 2, sh / 2)]
    j = rng.uniform(.02, .09)
    dst = []
    for (x, y) in esquinas:
        x += rng.uniform(-j, j) * sw
        y += rng.uniform(-j, j) * sh
        dst.append((cx + x * math.cos(ang) - y * math.sin(ang), cy + x * math.sin(ang) + y * math.cos(ang)))
    src = [(0, 0), (dw, 0), (dw, dh), (0, dh)]
    co = _coefs_perspectiva(dst, src)
    warped = doc.transform((FW, FH), Image.PERSPECTIVE, co, Image.BICUBIC)
    mask = Image.new("L", doc.size, 255).transform((FW, FH), Image.PERSPECTIVE, co, Image.BICUBIC)
    # sombra del papel sobre la superficie
    sombra = mask.filter(ImageFilter.GaussianBlur(12))
    off = (rng.randint(-8, 8), rng.randint(2, 14))
    fondo_arr = np.asarray(fondo).astype(float)
    sh_arr = np.zeros((FH, FW))
    sa = np.asarray(sombra).astype(float) / 255
    x0, y0 = max(off[0], 0), max(off[1], 0)
    sh_arr[y0:, x0:] = sa[: FH - y0, : FW - x0]
    fondo = Image.fromarray(np.clip(fondo_arr * (1 - .35 * sh_arr[..., None]), 0, 255).astype(np.uint8))
    fondo.paste(warped, (0, 0), mask)
    img = _iluminacion(rng, npr, fondo)
    if rng.random() < .5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(.3, 1.4)))
    if rng.random() < .15:  # movimiento
        k = [0] * 25
        for i in range(10, 15):
            k[i] = 1
        img = img.filter(ImageFilter.Kernel((5, 5), k, 5))
    return _ruido_jpeg(rng, npr, img, (1, 7), (40, 90))


def aug_escaneo(rng, npr, doc):
    """Simula un escaneo/fotocopia: leve rotación, gris, contraste, ruido, borde de escáner."""
    img = doc
    if rng.random() < .6:
        img = ImageOps.grayscale(img).convert("RGB")
    img = img.rotate(rng.uniform(-2.5, 2.5), resample=Image.BICUBIC, expand=True, fillcolor=rng.choice([(255, 255, 255), (60, 60, 60), (230, 230, 230)]))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(.8, 1.5))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(.9, 1.1))
    if rng.random() < .5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(.3, 1.0)))
    if rng.random() < .3:  # franja oscura de borde
        d = ImageDraw.Draw(img)
        w, h = img.size
        lado = rng.choice(["l", "r", "t"])
        k = rng.randint(5, 25)
        box = {"l": [0, 0, k, h], "r": [w - k, 0, w, h], "t": [0, 0, w, k]}[lado]
        d.rectangle(box, fill=(rng.randint(10, 60),) * 3)
    return _ruido_jpeg(rng, npr, img, (0, 5), (50, 92))


def aug_digital(rng, npr, doc):
    """PDF exportado / captura de pantalla: casi limpio, posible recorte y compresión."""
    img = doc
    w, h = img.size
    if rng.random() < .4:
        c = [rng.uniform(0, .04) * w, rng.uniform(0, .04) * h, w - rng.uniform(0, .04) * w, h - rng.uniform(0, .15) * h]
        img = img.crop([int(v) for v in c])
    esc = rng.uniform(.6, 1.0)
    img = img.resize((int(img.width * esc), int(img.height * esc)), Image.BICUBIC)
    return _ruido_jpeg(rng, npr, img, (0, 1.5), (65, 95))


MODOS = [("foto", .5, aug_foto), ("escaneo", .25, aug_escaneo), ("digital", .25, aug_digital)]


def en_hoja_a4(rng, doc):
    """Documento apaisado (media hoja) impreso arriba de una hoja A4 vertical. Se aplica igual a ambas clases."""
    w, h = doc.size
    H = int(w * 1.414)
    papel = doc.getpixel((2, h - 3))
    hoja = Image.new("RGB", (w, H), papel)
    hoja.paste(doc, (0, rng.randint(0, int(H * .05))))
    return hoja


def aumentar(rng, doc, modo=None):
    npr = np.random.default_rng(rng.randrange(2 ** 32))
    if doc.width > doc.height and rng.random() < .45:
        doc = en_hoja_a4(rng, doc)
    if modo is None:
        r, acum = rng.random(), 0
        for nombre, p, _ in MODOS:
            acum += p
            if r <= acum:
                modo = nombre
                break
    f = {m: fn for m, _, fn in MODOS}[modo]
    return f(rng, npr, doc), modo


def achicar(img, lado_max):
    w, h = img.size
    k = lado_max / max(w, h)
    if k < 1:
        img = img.resize((int(w * k), int(h * k)), Image.LANCZOS)
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def hoja_contacto(imgs, titulos, cols, ancho_celda, ruta):
    alto_celda = int(ancho_celda * 1.35)
    filas = math.ceil(len(imgs) / cols)
    hoja = Image.new("RGB", (cols * ancho_celda, filas * (alto_celda + 30)), (255, 255, 255))
    d = ImageDraw.Draw(hoja)
    f = _font(os.path.join(FUENTES["sans"][0][0], FUENTES["sans"][0][2]), 18)
    for i, (im, t) in enumerate(zip(imgs, titulos)):
        r, c = divmod(i, cols)
        im = im.copy()
        im.thumbnail((ancho_celda - 16, alto_celda - 8))
        x = c * ancho_celda + (ancho_celda - im.width) // 2
        y = r * (alto_celda + 30) + 30
        d.text((c * ancho_celda + 10, r * (alto_celda + 30) + 4), t, font=f, fill=(0, 0, 0))
        hoja.paste(im, (x, y))
        d.rectangle([x - 1, y - 1, x + im.width, y + im.height], outline=(180, 180, 180))
    hoja.save(ruta, quality=88)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--salida", default="datos_sinteticos")
    ap.add_argument("--n-pos", type=int, default=20, help="imágenes por familia de recibo")
    ap.add_argument("--n-neg-total", type=int, default=180, help="total de negativos (se reparte en partes iguales entre las 11 categorías)")
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--lado-max", type=int, default=1024, help="lado mayor de la imagen final, en px")
    ap.add_argument("--muestra", action="store_true", help="1 por familia/tipo, limpio + aumentado, con hojas de contacto")
    a = ap.parse_args()
    negs = NEGATIVOS
    pos = {k: [(k, fn)] for k, fn in FAMILIAS.items()}

    os.makedirs(a.salida, exist_ok=True)
    if a.muestra:
        rng = random.Random(a.semilla)
        limpios_p, aum_p, limpios_n, aum_n = [], [], [], []
        d_lim = os.path.join(a.salida, "limpios")
        d_aum = os.path.join(a.salida, "aumentados")
        os.makedirs(d_lim, exist_ok=True)
        os.makedirs(d_aum, exist_ok=True)
        modos = ["foto", "escaneo", "digital"]
        items = [(1, c, st, fn) for c, vs in pos.items() for st, fn in vs] + [(0, c, st, fn) for c, vs in negs.items() for st, fn in vs]
        tit_p, tit_n = [], []
        for i, (et, cat, sub, fn) in enumerate(items):
            doc = fn(random.Random(a.semilla * 1000 + i))
            aug, modo = aumentar(random.Random(a.semilla * 7 + i), doc, modos[i % 3] if i % 2 == 0 else "foto")
            pref = "pos" if et else "neg"
            nombre = cat if cat == sub else f"{cat}-{sub}"
            doc.save(os.path.join(d_lim, f"{pref}_{nombre}.png"))
            achicar(aug, a.lado_max).save(os.path.join(d_aum, f"{pref}_{nombre}_{modo}.jpg"), quality=90)
            if et:
                limpios_p.append(doc); aum_p.append((aug, f"{nombre} ({modo})")); tit_p.append(nombre)
            else:
                limpios_n.append(doc); aum_n.append((aug, f"{nombre} ({modo})")); tit_n.append(nombre)
        hoja_contacto(limpios_p, tit_p, 3, 560, os.path.join(a.salida, "hoja_positivos_limpios.jpg"))
        hoja_contacto(limpios_n, tit_n, 4, 480, os.path.join(a.salida, "hoja_negativos_limpios.jpg"))
        hoja_contacto([x for x, _ in aum_p], [t for _, t in aum_p], 3, 560, os.path.join(a.salida, "hoja_positivos_aumentados.jpg"))
        hoja_contacto([x for x, _ in aum_n], [t for _, t in aum_n], 4, 480, os.path.join(a.salida, "hoja_negativos_aumentados.jpg"))
        print(f"Muestra guardada en {a.salida}")
        return

    dp, dn = os.path.join(a.salida, "positivos"), os.path.join(a.salida, "negativos")
    os.makedirs(dp, exist_ok=True)
    os.makedirs(dn, exist_ok=True)
    filas = []
    k = 0
    base, resto = divmod(a.n_neg_total, len(negs))
    cuotas_neg = {k: base + (1 if j < resto else 0) for j, k in enumerate(negs)}
    for etiqueta, grupo, carpeta in [(1, pos, dp), (0, negs, dn)]:
        for nombre, variantes in grupo.items():
            n = a.n_pos if etiqueta else cuotas_neg[nombre]
            for i in range(n):
                semilla = a.semilla * 100000 + k
                k += 1
                rng = random.Random(semilla)
                subtipo, fn = variantes[i % len(variantes)]  # variantes repartidas en partes iguales
                doc = fn(rng)
                img, modo = aumentar(rng, doc)
                img = achicar(img, a.lado_max)
                archivo = f"{nombre}_{i:03d}.jpg"
                img.save(os.path.join(carpeta, archivo), quality=90)
                filas.append(dict(archivo=os.path.join(os.path.basename(carpeta), archivo), etiqueta=etiqueta,
                                  clase="recibo" if etiqueta else "no_recibo", familia=nombre, subtipo=subtipo,
                                  modo_aug=modo, semilla=semilla))
            print(f"  {'+' if etiqueta else '-'} {nombre}: {n}")
    with open(os.path.join(a.salida, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"Listo: {len(filas)} imágenes en {a.salida}")


if __name__ == "__main__":
    main()
