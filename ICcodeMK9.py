import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import os 
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import CoolProp.CoolProp as CP

# Os pontos de 1 a 4 do ciclo sao modelados ate a seçao 5, onde o metodo de Runge-Kutta é 
# implementado para resolver as EDOs, os pontos 5, "e", e "i" sao modelados na seçao 5.1

# ------------------------------ ESTE É O CÓDIGO USADO NO ENCIT E CIC ! -------------------------#
#------------------------------------------Theo Coelho Dias--------------------------------------#

# ==== Wiebe Dupla + Gamma Variavel (Testing!) + AFR FIXED! + Pressoes FIXED! + nth FIXED! + Metricas de erro ==== #

# =================================================================
# 1. INPUTS E CONFIGURAÇÃO (Interface do Usuário)
# =================================================================

COMBUSTIVEL_ESCOLHIDO = 'diesel'  # Opções: 'etanol', 'gasolina', 'diesel', 'metanol', 'hidrogenio' 
AFR_ESPECIFICO = 27.0             # Se None, usa o AFR estequiométrico do combustível escolhido 

# --- PARAMETROS DE COLETOR ---
P1 = 0.80                     # Pressão de admissão [bar]
P5 = 1.25                     # Pressão de escape [bar]
Ti_COLETOR = 298              # Temperatura do coletor [K]

# Geometria do motor base
geometria_motor = {'r': 20,     # Razão de compressão
                   'b': 0.078,  # Diâmetro do cilindro [m] (bore)
                   's': 0.062}  # Curso do pistão [m] (stroke)

R_ar = 287                    # Constante dos gases para o ar [J/kg K] 
eta_combustao = 0.9           # Eficiência de combustão (0 a 1) (para diesel em carga media fica entre 0.98 e 0.995)
h = 0.5                       # Passo de integração em graus (menor = mais preciso, mas mais lento)
f = 0.01                      # Fração residual de combustível na mistura (0 a 1, onde 1 é 100% combustível)
Te = 900.0                    # Temperatura efetiva dos gases de escape [K] (usada na convergência)

# -------------------------------
# BANCO DE DADOS DE COMBUSTÍVEIS
# -------------------------------
BANCO_DE_COMBUSTIVEIS = {
    'etanol': {
        'pci': 26.8e6,
        'afr_esteq': 9.0,
        'gamma': 1.33
    },
    'gasolina': {
        'pci': 44.4e6,
        'afr_esteq': 14.7,
        'gamma': 1.32
    },
    'diesel': {
        'pci': 42.5e6,
        'afr_esteq': 14.5,
        'gamma': 1.35
    },
    'metanol': {
        'pci': 19.7e6,
        'afr_esteq': 6.4,
        'gamma': 1.33
    },
    'hidrogenio': {
        'pci': 120.0e6,
        'afr_esteq': 34.3,
        'gamma': 1.38
    }
}

# =================================================================
# FUNÇÃO PARA CALCULAR GAMMA VIA COOLPROP (MISTURA AR + COMBUSTÍVEL)
# =================================================================
def get_gamma(T, P, AFR, fluido_comb='n-Dodecane', diag=None):
    """
    Calcula gamma (Cp/Cv) dinamicamente usando CoolProp para a mistura real.
    Usa frações mássicas para ponderar as capacidades térmicas do Ar e do Combustível.

    diag: dicionário opcional {'total': int, 'fallback': int, 'T_fallback': []}
          usado para diagnosticar quantas vezes o CoolProp falhou e caiu no
          valor fixo de segurança (1.35), e em que temperaturas isso ocorreu.
    """
    if diag is not None:
        diag['total'] = diag.get('total', 0) + 1

    T_clip = max(T, 300.0)    # Limite inferior de temperatura [K]
    T_clip = min(T_clip, 1000.0)   # Limite superior de temperatura [K] (limite do n-Dodecano)
    P_clip = max(P, 10000.0)  # Limite inferior de pressão [Pa]

    # Diagnóstico de CLIPPING (diferente de falha/exceção): registra quando o
    # T ou P reais do ciclo foram truncados antes de entrar no CoolProp, ou
    # seja, o gamma retornado não corresponde ao estado termodinâmico real
    # simulado naquele ponto, mesmo que o CoolProp não tenha lançado erro.
    if diag is not None and (T != T_clip or P != P_clip):
        diag['clipped'] = diag.get('clipped', 0) + 1
        diag.setdefault('T_clipped', []).append(T)
        if T > 1000.0:
            diag['clipped_alta_T'] = diag.get('clipped_alta_T', 0) + 1
        if T < 300.0:
            diag['clipped_baixa_T'] = diag.get('clipped_baixa_T', 0) + 1

    try:
        # 1. Frações Mássicas da mistura
        Y_ar = AFR / (AFR + 1)
        Y_comb = 1 / (AFR + 1)

        # 2. Capacidades Térmicas do Ar Puro
        Cp_ar = CP.PropsSI('C', 'T', T_clip, 'P', P_clip, 'Air')
        Cv_ar = CP.PropsSI('CVMASS', 'T', T_clip, 'P', P_clip, 'Air')

        # 3. Capacidades Térmicas do Combustível (n-Dodecano)
        Cp_comb = CP.PropsSI('C', 'T', T_clip, 'P', P_clip, fluido_comb)
        Cv_comb = CP.PropsSI('CVMASS', 'T', T_clip, 'P', P_clip, fluido_comb)

        # 4. Propriedades da Mistura (Gás Ideal)
        Cp_mix = Y_ar * Cp_ar + Y_comb * Cp_comb
        Cv_mix = Y_ar * Cv_ar + Y_comb * Cv_comb

        return Cp_mix / Cv_mix
    except Exception as e:
        # Em caso de falha de convergência do CoolProp, volta para o valor fixo de segurança
        if diag is not None:
            diag['fallback'] = diag.get('fallback', 0) + 1
            diag.setdefault('T_fallback', []).append(T)
            diag.setdefault('erros', {})
            msg = str(e)
            diag['erros'][msg] = diag['erros'].get(msg, 0) + 1
        return 1.35

def obter_dados_combustivel(nome_combustivel):
    """
    Busca as propriedades de um combustível no banco de dados.
    """
    nome_limpo = nome_combustivel.lower().strip()
    if nome_limpo in BANCO_DE_COMBUSTIVEIS:
        return BANCO_DE_COMBUSTIVEIS[nome_limpo]
    else:
        print(f"Erro: Combustível '{nome_combustivel}' não encontrado no banco de dados.")
        print(f"Opções disponíveis: {list(BANCO_DE_COMBUSTIVEIS.keys())}")
        return None

def calcular_dados_combustao(PCI, AFR, Pi, Ti, r, s, b, R_ar, eta_combustao):
    """
    Calcula o Q_total e as massas. recebe o PCI como argumento.
    """
    try:
        Vd = (s * b**2 * np.pi) / 4
        V1 = (Vd * r) / (r - 1)
    except ZeroDivisionError:
        return None
        
    m_ar = (Pi * V1) / (R_ar * Ti)
    m_fuel = m_ar / AFR
    m_total = m_ar + m_fuel
    Q_total = m_fuel * PCI * eta_combustao
    
    print(f"----- Cálculo de Combustão -----")
    print(f"  A/F: {AFR:.1f}:1")
    print(f"  Massa de Ar: {m_ar * 1000:.4f} g/ciclo") 
    print(f"  Massa de Comb.: {m_fuel * 1000:.4f} g/ciclo")
    print(f"  Q_total Calculado: {Q_total:.2f} J/ciclo")
    print("---------------------------------")
    
    return Q_total, m_total

# -------------------------------
# 2 FUNÇÃO DE WIEBE (WIEBE DUPLA - DIESEL) E DERIVADA
# -------------------------------
def dx_dtheta(theta, params):
    theta0 = params.get('theta0', 4)
    
    delta_p = params.get('delta_p', 11.5)
    a_p = params.get('a_p', 7.65)
    n_p = params.get('n_p', 3.13)
    
    delta_d = params.get('delta_d', 65.0)
    a_d = params.get('a_d', 3.64)
    n_d = params.get('n_d', 1.19)
    
    beta = params.get('beta_wiebe', 0.251)
    
    if theta < theta0:
        return 0.0

    if delta_p > 0 and theta <= theta0 + delta_p:
        phi_p = (theta - theta0) / delta_p
        if phi_p == 0 and (n_p - 1) <= 0:
            dx_pre = 0.0
        else:
            dx_pre = (a_p * n_p / delta_p) * (phi_p**(n_p - 1)) * np.exp(-a_p * (phi_p**n_p))
    else:
        dx_pre = 0.0

    if delta_d > 0 and theta <= theta0 + delta_d:
        phi_d = (theta - theta0) / delta_d
        if phi_d == 0 and (n_d - 1) <= 0:
             dx_dif = 0.0
        else:
             dx_dif = (a_d * n_d / delta_d) * (phi_d**(n_d - 1)) * np.exp(-a_d * (phi_d**n_d))
    else:
        dx_dif = 0.0
        
    return beta * dx_pre + (1 - beta) * dx_dif

# -------------------------------
# 3 VOLUME E DERIVADA (Eq.2.38 Ferguson)
# -------------------------------
def V(theta, params):
    theta_rad = np.radians(theta)
    V1 = params['V1']
    r = params['r']
    return (V1/r + (V1/(2*r)) * (r - 1) * (1 - np.cos(theta_rad)))

def dV_dtheta(theta, params):
    theta_rad = np.radians(theta)
    V1 = params['V1']
    r = params['r']
    return ((V1/(2*r)) * (r - 1) * np.sin(theta_rad) * (np.pi / 180))

# -------------------------------
# 4 EQUAÇÕES DIFERENCIAIS
# — GAMMA VARIÁVEL: usa interpolador se disponível, senão usa valor fixo
# -------------------------------
def calculate_derivatives(theta, Y, params):
    P = Y[0]
    V_theta = V(theta, params)
    dV_theta = dV_dtheta(theta, params)
    dx = dx_dtheta(theta, params)
    Q_total = params['Q_total']

    # ---- GAMMA DINÂMICO ----
    # Se 'gamma_func' existir em params, é um interpolador construído a partir
    # do vetor de temperaturas da 1ª passagem. Caso contrário, usa gamma fixo.
    if 'gamma_func' in params:
        gamma = float(params['gamma_func'](theta))
    else:
        gamma = params['gamma']
    # ------------------------

    if V_theta == 0:
        dp_dtheta = 0.0
        dw_dtheta = 0.0
    else:
        termo1_p = -(gamma * P / V_theta) * dV_theta
        termo2_p = ((gamma - 1) * Q_total / V_theta) * dx
        dp_dtheta = termo1_p + termo2_p
        dw_dtheta = P * dV_theta
    
    return np.array([dp_dtheta, dw_dtheta])

# -------------------------------
# 5 MÉTODO DE RUNGE-KUTTA 4ª ORDEM
# -------------------------------
def solve_cycle(P0, W0, theta_i, theta_f, h, params):
    try:
        if 'V1' not in params:
             params['Vd'] = (params['s'] * params['b']**2 * np.pi) / 4
             params['V1'] = (params['Vd'] * params['r']) / (params['r'] - 1)
        m_total = params['m_total']
    except ZeroDivisionError:
        print("Erro: Razão de compressão 'r' não pode ser 1.")
        return None
    except KeyError as e:
        print(f"Erro: Parâmetro {e} não encontrado.")
        return None

    thetas = [theta_i]
    Y = np.array([P0, W0])
    Ps_list = [P0]
    Ws_list = [W0]

    theta = theta_i
    num_steps = int((theta_f - theta_i) / h)
    
    for _ in range(num_steps):
        k1 = calculate_derivatives(theta, Y, params)
        k2 = calculate_derivatives(theta + h/2, Y + h*k1/2, params)
        k3 = calculate_derivatives(theta + h/2, Y + h*k2/2, params)
        k4 = calculate_derivatives(theta + h, Y + h*k3, params)

        Y += (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        theta += h

        thetas.append(theta)
        Ps_list.append(Y[0]) 
        Ws_list.append(Y[1])

    thetas_np = np.array(thetas)
    Ps_Pa_np = np.array(Ps_list)
    Ws_np = np.array(Ws_list)
    W_net = Ws_np[-1]
    
    return {
        'thetas': thetas_np,
        'Ps_Pa': Ps_Pa_np,
        'Ws': Ws_np,
        'W_net_J': W_net,
        'eta_th_percent': 0.0,
        'imep_bar': 0.0,
        'Vd_m3': params.get('Vd', 0),
        'V1_m3': params.get('V1', 0),
        'm_total': m_total
    }

# =================================================================
# 5.0 RK4 COM GAMMA VARIÁVEL (2 PASSAGENS)
# =================================================================
def solve_cycle_gamma_variavel(P0, W0, theta_i, theta_f, h, params):
    """
    Resolve o ciclo com gamma variável em 2 passagens:

    PASSAGEM 1 — gamma fixo (valor do banco de dados):
        Roda o RK4 normalmente com gamma constante.
        O objetivo é obter uma estimativa do vetor de temperatura T(theta)
        ao longo do ciclo de compressão e expansão.

    CONSTRUÇÃO DO INTERPOLADOR:
        Com o vetor T(theta) estimado, calcula gamma(T, P) em cada ponto
        usando CoolProp (n-Dodecano como surrogate do diesel).
        Em seguida, cria uma função interpoladora gamma(theta) contínua,
        que o RK4 pode consultar a qualquer ângulo theta.

    PASSAGEM 2 — gamma dinâmico (via interpolador):
        Roda o RK4 novamente, mas agora em cada passo o gamma é lido
        do interpolador — ou seja, ele varia com a temperatura local
        estimada na passagem anterior.

    O resultado final (res_pot) é a solução com gamma fisicamente consistente.
    """
    print("  [γ variável] Passagem 1: estimando T(θ) com γ fixo...")

    # --- PASSAGEM 1: gamma fixo ---
    res1 = solve_cycle(P0, W0, theta_i, theta_f, h, params)

    # Calcula T estimada em cada ponto do ciclo
    thetas_1 = res1['thetas']
    P_1      = res1['Ps_Pa']
    V_1      = V(thetas_1, params)
    T_1      = (P_1 * V_1) / (params['m_total'] * params['R'])

    print("[γ variável] Calculando γ(T,P) da Mistura via CoolProp para cada ponto...")

    # Dicionário de diagnóstico: conta quantas chamadas ao CoolProp caíram no
    # fallback (1.35) por estourar a faixa de validade do n-Dodecano (ou outro
    # erro), e em que temperaturas isso aconteceu.
    diag_coolprop = {}

    # Calcula gamma da MISTURA em cada ponto usando CoolProp
    gamma_vec = np.array([
        get_gamma(T, P, params['AFR'], diag=diag_coolprop)
        for T, P in zip(T_1, P_1)
    ])

    total = diag_coolprop.get('total', 0)
    n_fallback = diag_coolprop.get('fallback', 0)
    pct_fallback = (n_fallback / total * 100) if total > 0 else 0.0
    print(f"  [γ variável] CoolProp OK (sem exceção) em {total - n_fallback}/{total} pontos "
          f"({100 - pct_fallback:.1f}%) | Fallback por exceção (γ=1.35) em {n_fallback}/{total} "
          f"pontos ({pct_fallback:.1f}%)")
    if n_fallback > 0:
        T_fb = np.array(diag_coolprop.get('T_fallback', []))
        print(f"  [γ variável] Faixa de T nos pontos de fallback: "
              f"{T_fb.min():.0f}–{T_fb.max():.0f} K")
        for msg, count in diag_coolprop.get('erros', {}).items():
            print(f"  [γ variável]   {count}x: {msg}")

    n_clip = diag_coolprop.get('clipped', 0)
    pct_clip = (n_clip / total * 100) if total > 0 else 0.0
    n_clip_alta = diag_coolprop.get('clipped_alta_T', 0)
    n_clip_baixa = diag_coolprop.get('clipped_baixa_T', 0)
    print(f"  [γ variável] CLIPPING de T/P (sem exceção, mas T real ≠ T usada no CoolProp): "
          f"{n_clip}/{total} pontos ({pct_clip:.1f}%)")
    if n_clip > 0:
        T_clipped = np.array(diag_coolprop.get('T_clipped', []))
        print(f"  [γ variável]   -> {n_clip_alta} pontos com T > 1000 K (clipados para 1000 K)")
        print(f"  [γ variável]   -> {n_clip_baixa} pontos com T < 300 K (clipados para 300 K)")
        print(f"  [γ variável]   -> Faixa real de T nos pontos clipados: "
              f"{T_clipped.min():.0f}–{T_clipped.max():.0f} K")

    gamma_medio = np.mean(gamma_vec)
    gamma_min   = np.min(gamma_vec)
    gamma_max   = np.max(gamma_vec)
    print(f"  [γ variável] γ médio={gamma_medio:.4f} | min={gamma_min:.4f} | max={gamma_max:.4f}")

    # Cria interpolador gamma(theta) — usado dentro do RK4 na 2ª passagem
    gamma_interp = interp1d(thetas_1, gamma_vec,
                            kind='linear',
                            fill_value='extrapolate')

    print("  [γ variável] Passagem 2: RK4 com γ dinâmico...")

    # --- PASSAGEM 2: gamma dinâmico ---
    params_2a = params.copy()
    params_2a['gamma_func'] = gamma_interp  # injeta o interpolador nos params

    res2 = solve_cycle(P0, W0, theta_i, theta_f, h, params_2a)

    # Guarda os vetores de gamma e o diagnóstico do CoolProp para plotagem/relatório
    res2['gamma_vec']   = gamma_vec
    res2['thetas_gamma']= thetas_1
    res2['diag_coolprop'] = diag_coolprop

    print("  [γ variável] Concluído.\n")
    return res2

# -------------------------------
# 5.1 ADMISSÃO/ESCAPE E 5.2 (CONVERGÊNCIA)
# -------------------------------
def simular_admissao(params, P_intake, T_inicial, T_final, theta_start=-360, theta_end=-180, step=0.5):
    thetas = np.arange(theta_start, theta_end, step)
    Vs = V(thetas, params)
    Ps = np.full_like(thetas, P_intake, dtype=float)
    Ts = np.linspace(T_inicial, T_final, len(thetas))
    return {'thetas': thetas, 'Ps_Pa': Ps, 'Vs_m3': Vs, 'Ts_K': Ts}

def simular_escape(params, P_exhaust, P4, T4, theta_start=180, theta_end=360, step=0.5):
    thetas = np.arange(theta_start, theta_end + step, step)
    Vs = V(thetas, params)
    decay = 0.15
    theta_norm = thetas - theta_start
    Ps = (P4 - P_exhaust) * np.exp(-decay * theta_norm) + P_exhaust
    Ts = (T4 - 500) * np.exp(-decay * theta_norm) + 500
    return {'thetas': thetas, 'Ps_Pa': Ps, 'Vs_m3': Vs, 'Ts_K': Ts}

def calcular_convergencia(params_base, Pi, Pe, Ti, tol=1e-3, max_iter=30):
    params = params_base.copy()
    Te = 900
    f  = 0.01
    r = params['r']; s = params['s']; b = params['b']
    params['Vd'] = (s * b**2 * np.pi) / 4
    params['V1'] = (params['Vd'] * r) / (r - 1)

    gamma = params['gamma']; R = params['R']
    V1 = params['V1']; PCI = params['PCI']; AFR = params['AFR']
    
    print(f"--- Iniciando Convergência T1 e f (r={r}) ---")
    for i in range(max_iter):
        term_p = 1 - (Pi / Pe)
        term_g = (gamma - 1) / gamma
        colchete = 1 - (term_g * term_p)
        T1 = (1 - f) * Ti + f * colchete * Te
        
        m_total = (Pi * V1) / (R * T1)
        params['m_total'] = m_total
        params['Ti'] = T1
        
        m_mistura = m_total * (1 - f)
        m_fuel = m_mistura / (AFR + 1)
        params['Q_total'] = m_fuel * PCI * params['eta_combustao']
        
        res = solve_cycle(Pi, 0.0, -180, 180, 0.5, params)
        if res is None: break
        
        P4 = res['Ps_Pa'][-1]
        T4 = (P4 * V1) / (m_total * R)
        
        try:
            Te_new = T4 * (Pe / P4)**((gamma-1)/gamma)
            f_new  = (1/r) * (Pe/P4)**(1/gamma)
        except:
            Te_new = T4; f_new = f
            
        err_f = abs(f_new - f)
        f = f_new; Te = Te_new
        if err_f < tol:
            return params, T1, f, Te, P4, T4
            
    return params, T1, f, Te, P4, T4

# -------------------------------
# 5.2 MÉTRICAS DE ERRO (VALIDAÇÃO NUMÉRICA SIMULADO x EXPERIMENTAL)
# -------------------------------
def calcular_metricas_erro(x_exp, y_exp, x_sim, y_sim, nome_metrica="Curva"):
    """
    Compara uma curva simulada (x_sim, y_sim) com uma curva experimental (x_exp, y_exp).
    x_sim precisa estar ordenado de forma crescente (é o caso do vetor 'thetas' concatenado).

    Retorna um dicionário com R², RMSE, MAE, MAPE, erro no pico (%) e erro de fase do pico (°).
    """
    x_exp = np.asarray(x_exp, dtype=float)
    y_exp = np.asarray(y_exp, dtype=float)

    # Interpola a curva simulada nos MESMOS ângulos em que existe dado experimental
    y_sim_interp = np.interp(x_exp, x_sim, y_sim)

    residuos = y_exp - y_sim_interp
    ss_res = np.sum(residuos ** 2)
    ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    rmse = np.sqrt(np.mean(residuos ** 2))
    mae = np.mean(np.abs(residuos))

    mask_validos = np.abs(y_exp) > 1e-6
    mape = np.mean(np.abs(residuos[mask_validos] / y_exp[mask_validos])) * 100 if mask_validos.any() else np.nan

    idx_pico_exp = np.argmax(y_exp)
    idx_pico_sim = np.argmax(y_sim)
    pico_exp = y_exp[idx_pico_exp]
    pico_sim = y_sim[idx_pico_sim]
    theta_pico_exp = x_exp[idx_pico_exp]
    theta_pico_sim = x_sim[idx_pico_sim]

    erro_pico_pct = (pico_sim - pico_exp) / pico_exp * 100
    erro_fase_pico = theta_pico_sim - theta_pico_exp

    print(f"\n--- Error Metrics: {nome_metrica} ---")
    print(f"  R²:                  {r2:.4f}")
    print(f"  RMSE:                {rmse:.3f}")
    print(f"  MAE:                 {mae:.3f}")
    print(f"  MAPE:                {mape:.2f} %")
    print(f"  Experimental peak:   {pico_exp:.2f} @ {theta_pico_exp:.1f}°")
    print(f"  Simulated peak:      {pico_sim:.2f} @ {theta_pico_sim:.1f}°")
    print(f"  Erro no Pico:        {erro_pico_pct:+.2f} %")
    print(f"  Erro de Fase (pico): {erro_fase_pico:+.1f}°")
    print("-------------------------------------------")

    return {
        'r2': r2, 'rmse': rmse, 'mae': mae, 'mape_percent': mape,
        'pico_exp': pico_exp, 'theta_pico_exp': theta_pico_exp,
        'pico_sim': pico_sim, 'theta_pico_sim': theta_pico_sim,
        'erro_pico_percent': erro_pico_pct, 'erro_fase_pico_graus': erro_fase_pico
    }

# -------------------------------
# 6 DEFINIÇÃO DOS CENÁRIOS DE EXECUÇÃO
# -------------------------------

P1pa = P1 * 1e5
P5pa = P5 * 1e5

dados_combustivel = obter_dados_combustivel(COMBUSTIVEL_ESCOLHIDO)
if dados_combustivel is None: exit()
AFR_USADO = AFR_ESPECIFICO if AFR_ESPECIFICO else dados_combustivel['afr_esteq']

Q_calc, m_total_calc = calcular_dados_combustao(
    PCI=dados_combustivel['pci'], AFR=AFR_USADO,
    Pi=P1pa, Ti=Ti_COLETOR, r=geometria_motor['r'], s=geometria_motor['s'], b=geometria_motor['b'],
    R_ar=R_ar, eta_combustao=eta_combustao,
)

params_base = {
    **geometria_motor,
    'gamma': dados_combustivel['gamma'], 'R': R_ar,
    'PCI': dados_combustivel['pci'], 'AFR': AFR_USADO, 'eta_combustao': eta_combustao,
    'Ti': Ti_COLETOR, 'Q_total': Q_calc, 'm_total': m_total_calc,
    'theta0': 4.0,
    'delta_p': 11.6, 'a_p': 7.86, 'n_p': 3.14,
    'delta_d': 66.0, 'a_d': 3.76, 'n_d': 1.20,
    'beta_wiebe': 0.248
}

scenarios = {
    f"Base ({COMBUSTIVEL_ESCOLHIDO} r={params_base['r']})": params_base,
}

# -------------------------------
# 7 EXECUÇÃO, PLOTAGEM E COLETA DE DADOS
# -------------------------------

all_results_dfs = []
scenario_outputs = {} 

# --- Layout 2x2 para os gráficos ---
fig, ax = plt.subplots(2, 2, figsize=(18, 16))
fig.suptitle("Thermodynamic Cycle Analysis", fontsize=18)

# =================================================================
# LER DADOS EXPERIMENTAIS (COPIADOS E COLADOS)
# =================================================================
from meus_dados import dados_brutos_hrr
try:
    df_exp = pd.read_csv(io.StringIO(dados_brutos_hrr.strip()), sep='\t', decimal=',')
    angulos_exp = pd.to_numeric(df_exp['Crank Ang [deg]'], errors='coerce')
    hrr_exp = pd.to_numeric(df_exp['Q [kJ/m3deg] Diesel S10'], errors='coerce')
    
    mask = ~angulos_exp.isna() & ~hrr_exp.isna()
    angulos_exp = angulos_exp[mask]
    hrr_exp = hrr_exp[mask]
    
    print("Sucesso: Dados convertidos para números matemáticos puros!")
except Exception as e:
    print(f"Aviso: Erro ao ler os dados colados. Erro: {e}")
    angulos_exp, hrr_exp = [], []

ax1 = ax[0, 0] 
ax2 = ax[0, 1] 
ax3 = ax[1, 0] 
ax4 = ax[1, 1] 

# =================================================================
# EXTRAIR DADOS DE PRESSÃO X ÂNGULO 
# =================================================================
from meus_dados import dados_brutos_pressao

try:
    df_pressao = pd.read_csv(io.StringIO(dados_brutos_pressao.strip()), sep='\t', decimal=',')
    angulos_exp_pv = pd.to_numeric(df_pressao.iloc[:, 0], errors='coerce')
    pressao_exp_bar = pd.to_numeric(df_pressao.iloc[:, 1], errors='coerce')
    
    mask_ang_p = angulos_exp_pv.notna() & pressao_exp_bar.notna()
    angulos_exp_pv = angulos_exp_pv[mask_ang_p]
    pressao_exp_bar = pressao_exp_bar[mask_ang_p]
    
    print("Sucesso: Dados Brutos de Pressão x Ângulo lidos com perfeição!")
except Exception as e:
    print(f"Aviso: Falha ao ler os dados brutos de Pressão. Erro: {e}")

try:
    from meus_dados import dados_brutos_volume
    df_vol = pd.read_csv(io.StringIO(dados_brutos_volume.strip()), sep='\t', decimal=',', header=None)
    pv_volume = pd.to_numeric(df_vol.iloc[:, 0], errors='coerce')
    pv_pressao_bar = pd.to_numeric(df_vol.iloc[:, 1], errors='coerce')
    
    mask_pv = pv_pressao_bar.notna() & pv_volume.notna()
    pv_pressao_bar = pv_pressao_bar[mask_pv]
    pv_volume = pv_volume[mask_pv]
    
    if pv_volume.max() > 10.0:
        pv_volume_m3 = pv_volume / 1_000_000.0
    else:
        pv_volume_m3 = pv_volume
    print("Sucesso: Dados Brutos de Volume P-V lidos com perfeição!")

    theta_vol_full = np.arange(len(pv_pressao_bar)) - 360
except Exception as e:
    print(f"Aviso: Falha ao ler os dados brutos de Volume. Erro: {e}")
    pv_volume_m3 = []
    theta_vol_full = []

# Configurações Gráficos
ax1.set_title("Pressure vs. Crank Angle")
ax1.set_xlabel("Crank angle θ [°]")
ax1.set_ylabel("Cylinder pressure [bar]")
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

ax2.set_title("P-V Diagram")
ax2.set_xlabel("Cylinder volume [m³]")
ax2.set_ylabel("Cylinder pressure [bar]")
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

ax3.set_title("Temperature vs. Crank Angle")
ax3.set_xlabel("Crank angle θ [°]")
ax3.set_ylabel("Temperature [K]")
ax3.grid(True, which='both', linestyle='--', linewidth=0.5)

ax4.set_title("Heat Release Rate (HRR)")
ax4.set_xlabel("Crank angle θ [°]")
ax4.set_ylabel("Heat released [kJ/kg/°]")
ax4.set_xlim(-20, 60)

# Loop principal
for name, params_in in scenarios.items():
    print(f"Executando cenário: {name}...")
    
    # 1. ITERAÇÃO 
    dados_conv = calcular_convergencia(params_in, P1pa, P5pa, Ti_COLETOR)
    params_final, T1_res, f_res, Te_res, P4_res, T4_res = dados_conv
    
    # 2. GERAÇÃO DADOS (3 FASES)
    res_adm = simular_admissao(params_final, P1pa, Ti_COLETOR, T1_res, theta_end=-180, step=h)

    # ---- CORREÇÃO (Opção A) ----
    # Antes: res_pot = solve_cycle(P1pa, 0.0, -180, 180, h, params_final)
    # A função solve_cycle_gamma_variavel() já existia no código (CoolProp +
    # interpolador via 2 passagens de RK4), mas nunca era chamada — o gamma
    # variável ficava com código morto, e o ciclo de potência sempre rodava
    # com params_final['gamma'] fixo (1.35). Trocando para
    # solve_cycle_gamma_variavel(), o gamma dinâmico (via CoolProp,
    # n-Dodecano como surrogate) passa a de fato entrar no RK4 do curso de
    # potência (-180° a 180°, onde compressão/combustão/expansão ocorrem).
    res_pot = solve_cycle_gamma_variavel(P1pa, 0.0, -180, 180, h, params_final)
    
    theta_pot = res_pot['thetas']; P_pot = res_pot['Ps_Pa']
    V_pot = V(theta_pot, params_final)
    T_pot = (P_pot * V_pot) / (params_final['m_total'] * params_final['R'])
    
    res_esc = simular_escape(params_final, P5pa, P_pot[-1], T_pot[-1], theta_start=180, step=h)
    
    # 3. CONCATENAÇÃO
    thetas = np.concatenate([res_adm['thetas'], theta_pot, res_esc['thetas']])
    Ps_em_bar = np.concatenate([res_adm['Ps_Pa'], P_pot, res_esc['Ps_Pa']]) / 1e5
    Vs_m3 = np.concatenate([res_adm['Vs_m3'], V_pot, res_esc['Vs_m3']])
    Ts_K = np.concatenate([res_adm['Ts_K'], T_pot, res_esc['Ts_K']])
    
    # Cálculo Trabalho Total
    dV = np.diff(Vs_m3, prepend=Vs_m3[0])
    W_net = np.sum((Ps_em_bar * 1e5) * dV)
    
    # MUDE APENAS ESTA LINHA: Divide o trabalho pela energia química total real (os 413.24 J)
    eta_th   = (W_net / (params_final['Q_total'] / params_final['eta_combustao'])) * 100
    imep_bar = (W_net / params_final['Vd']) / 1e5
    
    # Encontra picos
    idx_max_P = np.argmax(Ps_em_bar)
    theta_max_P = thetas[idx_max_P]
    P_max = Ps_em_bar[idx_max_P]
    
    idx_max_T = np.argmax(Ts_K)
    theta_max_T = thetas[idx_max_T]
    T_max = Ts_K[idx_max_T]
    
    scenario_outputs[name] = {
        'P_max_bar': P_max, 'theta_max_P_graus': theta_max_P,
        'T_max_K': T_max, 'theta_max_T_graus': theta_max_T, 
        'W_net_J': W_net, 
        'eta_th_percent': eta_th,
        'imep_bar': imep_bar,
        'Q_total': params_final['Q_total'], 'V1': params_final['V1'],
        'gamma_vec': res_pot.get('gamma_vec'),
        'thetas_gamma': res_pot.get('thetas_gamma'),
        'diag_coolprop': res_pot.get('diag_coolprop'),
    }

    # -------------------------------------------------------------
    # 7.1 MÉTRICAS DE ERRO: PRESSÃO x ÂNGULO (janela -123° a 155°)
    # -------------------------------------------------------------
    metricas_pressao_janela = None
    if 'angulos_exp_pv' in locals() and len(angulos_exp_pv) > 0:
        metricas_pressao_janela = calcular_metricas_erro(
            x_exp=angulos_exp_pv.to_numpy(),
            y_exp=pressao_exp_bar.to_numpy(),
            x_sim=thetas,
            y_sim=Ps_em_bar,
            nome_metrica=f"Pressão x Ângulo (janela exp., {name})"
        )
        scenario_outputs[name]['metricas_pressao_janela'] = metricas_pressao_janela

    # -------------------------------------------------------------
    # 7.2 MÉTRICAS DE ERRO: PRESSÃO x ÂNGULO (CICLO COMPLETO 720°)
    #     usa os dados reconstruídos de dados_brutos_volume
    # -------------------------------------------------------------
    metricas_pressao_completo = None
    if 'theta_vol_full' in locals() and len(theta_vol_full) > 0:
        metricas_pressao_completo = calcular_metricas_erro(
            x_exp=np.asarray(theta_vol_full, dtype=float),
            y_exp=pv_pressao_bar.to_numpy(),
            x_sim=thetas,
            y_sim=Ps_em_bar,
            nome_metrica=f"Pressão x Ângulo (ciclo completo 720°, {name})"
        )
        scenario_outputs[name]['metricas_pressao_completo'] = metricas_pressao_completo
    
    label_plot = (
        f"{name}\n"
        f"  Peak P: {P_max:.1f} bar | Peak T: {T_max:.0f} K\n"
        f"  $\eta_{{th}}$: {eta_th:.1f}% | IMEP: {imep_bar:.1f} bar"
    )
    
    # PLOTAGEM
    cor_linha, = ax1.plot(thetas, Ps_em_bar, label=label_plot, linewidth=2)
    cor = cor_linha.get_color() 
    
    ax2.plot(Vs_m3, Ps_em_bar, linewidth=2, color=cor)
    ax3.plot(thetas, Ts_K, linewidth=2, color=cor)

    

# --- GRÁFICO 4: CÁLCULO E PLOTAGEM DO HRR EM kJ/kg/° ---
   
    dQ_dtheta_J = np.array([params_final['Q_total'] * dx_dtheta(t, params_final) for t in thetas])
    
    m_fuel_kg = params_final['Q_total'] / (params_final['PCI'] * params_final['eta_combustao'])
    dQ_dtheta_kJ_kg = (dQ_dtheta_J / m_fuel_kg) / 1000.0
    
    ax4.plot(thetas, dQ_dtheta_kJ_kg, linewidth=2, color=cor)
    ax4.fill_between(thetas, dQ_dtheta_kJ_kg, color=cor, alpha=0.1)
    
    theta_comb_start = params_final['theta0']
    maior_delta = max(params_final.get('delta_p', 0), params_final.get('delta_d', 0))
    theta_comb_end = params_final['theta0'] + maior_delta

    # PLOTA A CURVA EXPERIMENTAL SOBREPOSTA E CALCULA R2 DA MODELAGEM PRINCIPAL
    if len(angulos_exp) > 0:
        dQ_sim_exp = np.array([params_final['Q_total'] * dx_dtheta(t, params_final) for t in angulos_exp])
        dQ_sim_exp_kJ_kg = (dQ_sim_exp / m_fuel_kg) / 1000.0
        
        Vd_m3 = params_final['Vd']
        hrr_exp_np = np.array(hrr_exp) * (Vd_m3 / m_fuel_kg)
        
        ss_res = np.sum((hrr_exp_np - dQ_sim_exp_kJ_kg) ** 2)
        ss_tot = np.sum((hrr_exp_np - np.mean(hrr_exp_np)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        scenario_outputs[name]['R2_HRR'] = r2
        label_exp = f'Exp. (R²={r2:.4f} for {name.split()[0]})'
        
        ax4.plot(angulos_exp, hrr_exp_np, color='black', linestyle='--', linewidth=1.5, label=label_exp)
    
    ax4.legend(fontsize=9, loc='upper right')

    idx_start = np.argmin(np.abs(thetas - theta_comb_start))
    idx_end = np.argmin(np.abs(thetas - theta_comb_end))
    
    V_start_comb = Vs_m3[idx_start]; P_start_comb = Ps_em_bar[idx_start]
    V_end_comb = Vs_m3[idx_end]; P_end_comb = Ps_em_bar[idx_end]
    T_start_comb = Ts_K[idx_start]; T_end_comb = Ts_K[idx_end]

    ax2.annotate(f'Start: {P_start_comb:.1f} bar', xy=(V_start_comb, P_start_comb),
        xytext=(V_start_comb * 0.98, P_start_comb + 10),
        arrowprops=dict(color=cor, arrowstyle='->', shrinkA=5, shrinkB=1), fontsize=8, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=cor, alpha=0.7))
    ax2.plot(V_start_comb, P_start_comb, 'o', markersize=8, color=cor, markeredgecolor='k')

    ax2.annotate(f'End: {P_end_comb:.1f} bar', xy=(V_end_comb, P_end_comb),
        xytext=(V_end_comb * 1.02, P_end_comb + 10),
        arrowprops=dict(color=cor, arrowstyle='->', shrinkA=5, shrinkB=1), fontsize=8, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=cor, alpha=0.7))
    ax2.plot(V_end_comb, P_end_comb, 'X', markersize=8, color=cor, markeredgecolor='k')
    
    texto_inicio_T = f"Start: {theta_comb_start}°\n{T_start_comb:.0f} K"
    texto_fim_T = f"End: {theta_comb_end}°\n{T_end_comb:.0f} K"

    ax3.annotate(texto_inicio_T, xy=(theta_comb_start, T_start_comb),
        xytext=(theta_comb_start - 10, T_start_comb + 150), 
        arrowprops=dict(color=cor, arrowstyle='->', shrinkA=5, shrinkB=1), fontsize=8, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=cor, alpha=0.7))
    ax3.plot(theta_comb_start, T_start_comb, 'o', markersize=8, color=cor, markeredgecolor='k')

    ax3.annotate(texto_fim_T, xy=(theta_comb_end, T_end_comb),
        xytext=(theta_comb_end + 10, T_end_comb + 10), 
        arrowprops=dict(color=cor, arrowstyle='->', shrinkA=5, shrinkB=1), fontsize=8, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=cor, alpha=0.7))
    ax3.plot(theta_comb_end, T_end_comb, 'X', markersize=8, color=cor, markeredgecolor='k')

    df_run = pd.DataFrame({
        'theta_graus': thetas, 'pressao_bar': Ps_em_bar,
        'temperatura_K': Ts_K, 'volume_m3': Vs_m3, 'cenario': name
    })
    all_results_dfs.append(df_run)

# =================================================================
# PLOTAGEM EXPERIMENTAL - PRESSÃO E DIAGRAMA P-V
# =================================================================
if 'angulos_exp_pv' in locals() and len(angulos_exp_pv) > 0:
    ang_np = angulos_exp_pv.to_numpy()
    p_exp_np = pressao_exp_bar.to_numpy()
    
    ax1.plot(ang_np, p_exp_np, color='black', linestyle='--', linewidth=2, zorder=5, label='Exp. Pressure (Test Bench)')
    
    if 'pv_volume_m3' in locals() and len(pv_volume_m3) > 0:
        ax2.plot(pv_volume_m3.to_numpy(), pv_pressao_bar.to_numpy(), color='black', linestyle='--', linewidth=2, zorder=5, label='P-V Exp. (Test Bench Data)')
    else:
        V_exp_m3_calc = V(ang_np, params_final)
        ax2.plot(V_exp_m3_calc, p_exp_np, color='black', linestyle='--', linewidth=2, zorder=5, label='P-V Exp. (Test Bench Calc.)')

ax1.legend(fontsize=9, loc='upper left')
ax2.legend(fontsize=9, loc='upper right')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# -------------------------------
# 8 EXIBIÇÃO DE RESULTADOS 
# -------------------------------

if all_results_dfs:
    print("\n--- Resumo dos Outputs da Simulação ---")
    for name, data in scenario_outputs.items():
        print(f"\nCenário: {name}")
        Q_total_cenario = data['Q_total']
        P1 = P1pa 
        V1 = data['V1']
        Q_adim = Q_total_cenario / (P1 * V1)
        
        print(f"  Pico de Pressão:   {data['P_max_bar']:.2f} bar @ {data['theta_max_P_graus']:.1f}°")
        print(f"  Pico de Temperatura:{data['T_max_K']:.0f} K @ {data['theta_max_T_graus']:.1f}°")
        print(f"  Trabalho Líquido:   {data['W_net_J']:.2f} J")
        print(f"  Eficiência Térmica: {data['eta_th_percent']:.2f} %")
        print(f"  IMEP:               {data['imep_bar']:.2f} bar")
        print(f"  Q Adimensional:     {Q_adim:.2f}")
        if data.get('gamma_vec') is not None:
            gv = data['gamma_vec']
            print(f"  γ variável (γ médio/min/max): {np.mean(gv):.4f} / {np.min(gv):.4f} / {np.max(gv):.4f}")
        if data.get('diag_coolprop'):
            dc = data['diag_coolprop']
            tot = dc.get('total', 0); fb = dc.get('fallback', 0); clip = dc.get('clipped', 0)
            pct_fb = (fb/tot*100) if tot > 0 else 0.0
            pct_clip = (clip/tot*100) if tot > 0 else 0.0
            print(f"  CoolProp: {tot-fb}/{tot} pontos sem exceção ({100-pct_fb:.1f}%) | "
                  f"fallback γ=1.35 por exceção: {fb}/{tot} ({pct_fb:.1f}%) | "
                  f"pontos com T/P clipados antes do CoolProp: {clip}/{tot} ({pct_clip:.1f}%)")
        if 'R2_HRR' in data:
            print(f"  R² (TALC Cal. vs Exp): {data['R2_HRR']:.4f}")
        if 'metricas_pressao_completo' in data:
            m = data['metricas_pressao_completo']
            print(f"  R² Pressão (ciclo completo): {m['r2']:.4f} | RMSE: {m['rmse']:.3f} bar | MAE: {m['mae']:.3f} bar | MAPE: {m['mape_percent']:.2f}%")
            print(f"  Erro no pico de pressão: {m['erro_pico_percent']:+.2f}% | Erro de fase do pico: {m['erro_fase_pico_graus']:+.1f}°")
else:
    print("\nNenhum resultado de simulação foi gerado.")

# -------------------------------
# 9 OTIMIZAÇÃO DE PARÂMETROS WIEBE DUPLA (SCIPY CURVE FIT)
# -------------------------------
try:
    if len(angulos_exp) > 0 and 'params_final' in locals():
        print("\n--- Iniciando Otimização Dinâmica por Scipy Curve Fit ---")
        
        def funcao_objetivo(t_array, beta_wiebe, a_p, n_p, delta_p, a_d, n_d, delta_d, theta0):
            res = np.zeros_like(t_array, dtype=float)
            p_dict = params_final.copy()
            p_dict.update({
                'beta_wiebe': beta_wiebe, 'a_p': a_p, 'n_p': n_p, 'delta_p': delta_p,
                'a_d': a_d, 'n_d': n_d, 'delta_d': delta_d, 'theta0': theta0
            })
            for i, t in enumerate(t_array):
                res[i] = dx_dtheta(t, p_dict)
            
            m_fuel_kg = p_dict['Q_total'] / (p_dict['PCI'] * p_dict['eta_combustao'])
            return (res * p_dict['Q_total'] / m_fuel_kg) / 1000.0

        p0 = [0.15, 6.9, 4.0, 15.0, 6.9, 1.2, 50.0, -5.0]
        limite_inferior = [0.01,  0.1, 0.5,  2.0,  0.1, 0.1,  15.0, -30.0]
        limite_superior = [0.40, 10.0, 8.0, 30.0, 10.0, 5.0, 100.0,   5.0]
        
        m_fuel_kg_base = params_final['Q_total'] / (params_final['PCI'] * params_final['eta_combustao'])
        Vd_m3_base = params_final['Vd']
        hrr_exp_np = np.array(hrr_exp) * (Vd_m3_base / m_fuel_kg_base)
        ss_tot = np.sum((hrr_exp_np - np.mean(hrr_exp_np)) ** 2)
        
        popt, pcov = curve_fit(funcao_objetivo, angulos_exp, hrr_exp_np, p0=p0, bounds=(limite_inferior, limite_superior), maxfev=15000)
        
        dq_fit_kj = funcao_objetivo(angulos_exp, *popt)
        
        ss_res = np.sum((hrr_exp_np - dq_fit_kj) ** 2)
        r2_opt = 1 - (ss_res / ss_tot)
        
        print(f"Otimização Scipy concluída com sucesso!")
        print(f"Melhor R² Atingido = {r2_opt:.4f}")
        print("\nParâmetros Mapeados pelo Scipy:")
        print(f"  beta:    {popt[0]:.3f}")
        print(f"  a_p:     {popt[1]:.2f}    | n_p:     {popt[2]:.2f}    | delta_p: {popt[3]:.1f}°")
        print(f"  a_d:     {popt[4]:.2f}    | n_d:     {popt[5]:.2f}    | delta_d: {popt[6]:.1f}°")
        print(f"  theta0:  {popt[7]:.1f}°")
        
        fig2, ax_opt = plt.subplots(figsize=(10, 6))
        ax_opt.plot(angulos_exp, hrr_exp_np, 'k--', linewidth=2, label='Original Experimental Curve')
        ax_opt.plot(angulos_exp, dq_fit_kj, 'r-', linewidth=2.5, alpha=0.8, 
                    label=f'Optimized Curve (Scipy) | R² = {r2_opt:.4f}')
        ax_opt.set_title("Double Wiebe Function (Dynamic Least-Squares Optimization)", fontsize=14)
        ax_opt.set_xlabel("Crank Angle [°]")
        ax_opt.set_ylabel("Heat Released [kJ/kg/°]")
        ax_opt.set_xlim(-20, 60)
        ax_opt.grid(True, linestyle='--', alpha=0.7)
        ax_opt.legend()

        print("\n--- Gerando Heat Map (R²: beta_wiebe vs theta0) ---")
        beta_range = np.linspace(0.01, 0.40, 25)
        theta0_range = np.linspace(-15.0, 5.0, 25)
        
        B, T0 = np.meshgrid(beta_range, theta0_range)
        R2_grid = np.zeros_like(B)
        
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                dq_ij = funcao_objetivo(angulos_exp, B[i,j], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], T0[i,j])
                ss_res_ij = np.sum((hrr_exp_np - dq_ij) ** 2)
                R2_grid[i,j] = 1 - (ss_res_ij / ss_tot)
                
        fig3, ax_hm = plt.subplots(figsize=(8, 6))
        R2_plot = np.maximum(R2_grid, 0.0)
        cp = ax_hm.contourf(B, T0, R2_plot, levels=30, cmap='inferno')
        cbar = fig3.colorbar(cp)
        cbar.set_label('Coefficient of Determination (R²)', fontsize=11)
        
        ax_hm.plot(popt[0], popt[7], 'w*', markersize=14, markeredgecolor='k', label='Scipy Optimum')
        
        ax_hm.set_title("Heat Map: R² Surface (beta vs theta0)", fontsize=14)
        ax_hm.set_xlabel("Premixed Fraction (beta)", fontsize=12)
        ax_hm.set_ylabel("Start of Combustion (theta0) [°]", fontsize=12)
        ax_hm.legend()

except ImportError:
    print("Aviso: Biblioteca SciPy não está instalada.")
except Exception as e:
    print(f"Não foi possível rodar a otimização de Scipy Curve Fit. Erro: {e}")

# Exibe os gráficos
plt.show()