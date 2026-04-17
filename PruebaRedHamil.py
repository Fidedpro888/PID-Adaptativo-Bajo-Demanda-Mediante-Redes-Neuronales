# ZMQ + EKF-SNPID (freeze/histeresis) + referencias "sinusoidal" y "rosa" + gráficas
# + PREDICCIÓN DE HAMILTONIANO (comparación con H físico) con RED de 3 ENTRADAS [ux, uy, uth]
# + CAMBIO DE MASA DE CUBOID (m_L) EN TIEMPO DE SIMULACIÓN
# + DESCONGELAR GANANCIAS CUANDO |H_pred - H_real| EXCEDE UMBRAL
# + READAPTACIÓN EN LÍNEA (re-entrenamiento incremental) de la red con buffer FIFO

import time, math, sys, os
import numpy as np
import matplotlib.pyplot as plt

# ======================= Config carga (cuboid) =======================
CUBOID_PATH   = '/Cuboid'   # ruta del cubo en escena
ML_INIT       = 0.0         # masa inicial de carga
ML_NEW        = 10.0        # masa nueva de carga cuando "aparece"
ML_SWITCH_T   = None        # None => mitad de S; o poner manual un float (segundos)
ML_EPS        = 1e-6        # fallback para el motor

# =================== Config disparador por Hamiltoniano ===================
H_DIFF_ABS_THR = 0.5        # [J] umbral absoluto |H_pred - H_real|
H_DIFF_REL_THR = 0.25       # [adim] umbral relativo vs |H_real|
H_EPS_DEN      = 1e-6       # evita división por 0 en relativo
UNFREEZE_H_HOLD = 3.0       # [s] mantener descongelado tras disparo por H
PLOT_SWITCH_MARK = True     # marcar t_switch y eventos H en gráficas

# =================== Online learning (readaptación) ===================
ONLINE_BUFFER_SIZE = 512
ONLINE_BATCH_SIZE  = 128
ONLINE_STEPS_PER_TRIGGER = 32
ONLINE_LR = 1e-3
ONLINE_MIN_POINTS = 50
COOLDOWN_BETWEEN_TRIGGERS = 2.0  # [s]

# -------- Cliente ZMQ --------
try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    print("[ERROR] Instala:  pip install coppeliasim-zmqremoteapi-client")
    sys.exit(1)

# -------- TensorFlow --------
TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import backend as K
except Exception as e:
    print("[WARN] TensorFlow no disponible, solo se graficará H físico. Detalle:", e)
    TF_AVAILABLE = False

# === Modelo de 3 ENTRADAS ===
MODEL_PATH = "hamiltonian_tf_model.keras"  # Debe aceptar [ux, uy, uth]

def rmse_keras(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def r2_keras(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1.0 - ss_res / (ss_tot + K.epsilon())

def _load_tf_model_for_online(path):
    if not TF_AVAILABLE or not os.path.exists(path):
        print(f"[INFO] Modelo TF no encontrado o TF no disponible: {path}")
        return None
    try:
        print(f"[OK] Cargando modelo TF (safe_mode=False): {path}")
        model = keras.models.load_model(
            path,
            custom_objects={"rmse_keras": rmse_keras, "r2_keras": r2_keras},
            safe_mode=False
        )
    except TypeError:
        try:
            print("[INFO] Reintentando con enable_unsafe_deserialization()")
            keras.config.enable_unsafe_deserialization()
        except Exception:
            pass
        print(f"[OK] Cargando modelo TF (modo inseguro): {path}")
        model = keras.models.load_model(
            path,
            custom_objects={"rmse_keras": rmse_keras, "r2_keras": r2_keras}
        )
    except Exception as e:
        print("[WARN] Falló la carga del modelo TF. Detalle:", e)
        return None

    # Recompilar para entrenamiento en línea con LR pequeña
    model.compile(optimizer=keras.optimizers.Adam(ONLINE_LR), loss="mse")
    return model

def r2_score_np(y, yhat):
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1.0 - ss_res / (ss_tot + 1e-12)

# -------- Utilidades --------
def wrap_angle(a): return math.atan2(math.sin(a), math.cos(a))

def draw_robot_ax(ax, pose, L, l, color='C1'):
    x, y, th = pose
    pts_body = np.array([[ L,  0],[ 0,  l],[-L,  0],[ 0, -l],[ L,  0]], float).T
    Rm = np.array([[math.cos(th),-math.sin(th)],[math.sin(th),math.cos(th)]])
    pts_world = Rm @ pts_body + np.array([[x],[y]])
    ax.plot(pts_world[0], pts_world[1], color=color, lw=2)
    tip = np.array([x+0.25*math.cos(th), y+0.25*math.sin(th)])
    ax.arrow(x, y, tip[0]-x, tip[1]-y, head_width=0.05, length_includes_head=True, color=color)

# -------- Envoltura YouBot --------
class YouBot:
    def __init__(self, host='127.0.0.1', port=23000, wheel_radius=0.10, w_min=-2.5, w_max=2.5):
        self.r = wheel_radius
        self.w_min = np.full(4, w_min, float)
        self.w_max = np.full(4, w_max, float)
        print("[INFO] Conectando a CoppeliaSim (ZMQ)...")
        self.client = RemoteAPIClient(host=host, port=port)
        self.sim = self.client.getObject('sim')
        print("[OK] Conectado.")
        print("[INFO] Obteniendo handles...")
        self.youBot   = self.sim.getObject('/youBot')
        self.motor_fl = self.sim.getObject('/youBot/rollingJoint_fl')
        self.motor_fr = self.sim.getObject('/youBot/rollingJoint_fr')
        self.motor_rl = self.sim.getObject('/youBot/rollingJoint_rl')
        self.motor_rr = self.sim.getObject('/youBot/rollingJoint_rr')
        # Cubo (carga)
        try:
            self.cuboid = self.sim.getObject(CUBOID_PATH)
            print(f"[OK] Cuboid: {CUBOID_PATH}")
        except Exception as e:
            self.cuboid = None
            print(f"[WARN] No se encontró el cuboid en ruta {CUBOID_PATH}. Detalle:", e)
        print("[OK] Handles listos.")

    def get_pose(self):
        p = self.sim.getObjectPosition(self.youBot, -1)
        o = self.sim.getObjectOrientation(self.youBot, -1)  # [ax,ay,az]
        return np.array([float(p[0]), float(p[1]), float(o[2])], float)

    def set_wheel_velocities(self, w):
        w = np.asarray(w, float).flatten()
        w = np.maximum(w, self.w_min); w = np.minimum(w, self.w_max)
        # Mantengo signo negativo como tu MATLAB
        self.sim.setJointTargetVelocity(self.motor_fl, float(-w[0]))
        self.sim.setJointTargetVelocity(self.motor_fr, float(-w[1]))
        self.sim.setJointTargetVelocity(self.motor_rl, float(-w[2]))
        self.sim.setJointTargetVelocity(self.motor_rr, float(-w[3]))

    def stop(self): self.set_wheel_velocities([0,0,0,0])

# -------- EKF-SNPID (freeze/histeresis) --------
def ekf_snpid_freeze(x1,x2,x3, w,P,Q,R, n,alpha, umax, freeze_flag):
    x = np.array([x1,x2,x3], float)
    v = float(w @ x)
    u = alpha * np.tanh(v)
    u_sat = np.clip(u, -umax, umax)
    es    = u_sat - u
    e_aw  = x1 + es
    sech2 = 1.0 - np.tanh(v)**2
    H = (alpha * sech2) * x
    S = float(H @ (P @ H) + R)
    if freeze_flag:
        return u_sat, w.copy(), P.copy()
    Kf     = (P @ H) / S
    P_new  = (np.eye(3) - np.outer(Kf, H)) @ P + Q
    w_new  = w + (n * Kf) * e_aw
    return u_sat, w_new, P_new

# -------- Referencias ORIGINALES --------
def ref_original(t, modo):
    """
    xd, yd, thetad, modos: "sinusoidal" | "rosa"
    """
    if modo == "sinusoidal":
        xd  = 0.1 * t
        yd  = 0.4 * math.sin(0.5 * t)
        thd = -math.pi/8
        return xd, yd, thd
    elif modo == "rosa":
        a   = 0.2 + 0.05*math.cos(5*0.05*t)
        xd  = a * math.cos(0.05*t) - 0.25
        yd  = a * math.sin(0.05*t)
        thd = math.pi/4
        return xd, yd, thd
    else:
        raise ValueError("Modo de referencia no válido. Usa 'sinusoidal' o 'rosa'.")

# -------- Buffer de experiencias para readaptación --------
class OnlineBuffer:
    def __init__(self, capacity=ONLINE_BUFFER_SIZE):
        self.cap = capacity
        self.X = np.zeros((capacity,3), dtype=np.float32)
        self.y = np.zeros((capacity,1), dtype=np.float32)
        self.n = 0
        self.ptr = 0
    def push(self, ux, uy, uth, H):
        self.X[self.ptr] = (ux, uy, uth)
        self.y[self.ptr,0] = H
        self.ptr = (self.ptr + 1) % self.cap
        self.n = min(self.n + 1, self.cap)
    def sample(self, batch_size):
        if self.n == 0: return None, None
        idx = np.random.choice(self.n, size=min(batch_size, self.n), replace=False)
        return self.X[idx], self.y[idx]
    def ready(self, min_points=ONLINE_MIN_POINTS):
        return self.n >= min_points

# -------- Main --------
def main():
    # ===== Selección de referencia =====
    REF_MODO = "rosa"   # "sinusoidal" | "rosa"
    S        = 180.0    # duración [s]

    # ===== Geometría YouBot =====
    L = 0.1981; l = 0.1990

    # ===== EKF-SNPID params =====
    P_init = np.eye(3); Q = 0.1*np.eye(3); Rm = 1e-4
    n_x  = np.array([0.1, 0.1, 0.01])
    n_y  = np.array([0.1, 0.1, 0.01])
    n_th = np.array([0.1, 0.1, 0.01])
    alpha = 1.5
    uMax  = np.array([1.5, 1.5, 1.5])

    # Freeze por normalización (histeresis)
    e_thr_freeze=0.05; e_thr_unfz=0.10

    # Pesos iniciales
    w_x = np.array([0, 0, 0])
    w_y = np.array([0, 0, 0])
    w_t = np.array([0, 0, 0])
    
    w_x = np.array([0.8, 0, 0.06])
    w_y = np.array([6, 0.05, 0.02])
    w_t = np.array([1, 0.08, 0.02])
    
    # pesos de prueba
    # w_x = np.array([1.25, -0.05, 0.08])
    # w_y = np.array([8.0, 0.1, 0.2])
    # w_t = np.array([3.5, 0.001, 0.2])
    
    P_x = P_init.copy(); P_y = P_init.copy(); P_t = P_init.copy()

    # Integrales protegidas
    dx3=0.0; dy3=0.0; dt3=0.0
    freeze_x=False; freeze_y=False; freeze_t=False

    # ===== Conexión =====
    bot = YouBot(port=23000, wheel_radius=0.10, w_min=-2.5, w_max=2.5)
    bot.stop(); time.sleep(0.05)

    # ===== Carga (m_L) y masa del cuboid =====
    m_base=20.0; m_w=1.0; r_w=0.0475  # constantes

    m_L = ML_INIT
    if bot.cuboid is not None:
        try:
            bot.sim.setShapeMass(bot.cuboid, ML_INIT)
        except Exception as e:
            print("[WARN] El motor no aceptó masa=0. Uso epsilon.", e)
            bot.sim.setShapeMass(bot.cuboid, ML_EPS)
            m_L = ML_EPS
    else:
        print("[WARN] Sin cuboid en la escena: m_L será sólo interno.")

    # Tiempo de cambio de masa
    t_switch = (S*0.5) if (ML_SWITCH_T is None) else float(ML_SWITCH_T)
    mass_switched = False
    switch_time_logged = None

    # ===== Logs =====
    t_log=[]; p_log=[]; pd_log=[]; w_log=[]
    ux_log=[]; uy_log=[]; uth_log=[]
    ex_log=[]; ey_log=[]; eth_log=[]
    exn_log=[]; eyn_log=[]; ethn_log=[]
    Wx_log=[]; Wy_log=[]; Wt_log=[]; H_log=[]
    H_pred_list = []   # H predicho si hay modelo
    h_unfreeze_events = []  # tiempos cuando hubo unfreeze por H

    # ===== Preparar modelo TF (3 entradas) + buffer =====
    model = _load_tf_model_for_online(MODEL_PATH) if TF_AVAILABLE else None
    buffer = OnlineBuffer()
    last_trigger_time = -1e9
    hold_unfreeze_until = 0.0

    print(f"[INFO] EKF-SNPID con referencia '{REF_MODO}' (sin pausa).")
    t0 = time.time(); t_prev=None; p_prev=None;
    ex_prev=0.0; ey_prev=0.0; eth_prev=0.0

    try:
        while True:
            t = time.time() - t0
            if t > S: break

            # --- Referencia ORIGINAL ---
            xd, yd, thd = ref_original(t, REF_MODO)

            # --- Estado actual ---
            p = bot.get_pose(); xk, yk, th = p

            # --- Errores  ---
            ex  = xd - xk
            ey  = yd - yk
            eth = wrap_angle(thd - th)

            # Derivadas discretas
            dx1, dy1, dt1 = ex, ey, eth
            dx2, dy2, dt2 = ex-ex_prev, ey-ey_prev, eth-eth_prev

            # Integrales: sólo si hay avance
            if t_prev is not None and p_prev is not None:
                if np.linalg.norm(p - p_prev) > 1e-12:
                    dx3 += ex; dy3 += ey; dt3 += eth
            else:
                dx3 += ex; dy3 += ey; dt3 += eth

            # --- EKF-SNPID por eje ---
            ux,  w_x, P_x = ekf_snpid_freeze(dx1,dx2,dx3, w_x,P_x,Q,Rm, n_x, alpha,uMax[0], freeze_x)
            uy,  w_y, P_y = ekf_snpid_freeze(dy1,dy2,dy3, w_y,P_y,Q,Rm, n_y, alpha,uMax[1], freeze_y)
            uth, w_t, P_t = ekf_snpid_freeze(dt1,dt2,dt3, w_t,P_t,Q,Rm, n_th,alpha,uMax[2], freeze_t)

            # --- Freeze/unfreeze por normalización (con hold si aplica) ---
            exn = abs(ex); eyn = abs(ey); ethn = abs(eth)

            if t < hold_unfreeze_until:
                freeze_x = freeze_y = freeze_t = False
            else:
                if (not freeze_x) and (exn < e_thr_freeze): freeze_x=True
                elif freeze_x and (exn > e_thr_unfz):       freeze_x=False
                if (not freeze_y) and (eyn < e_thr_freeze): freeze_y=True
                elif freeze_y and (eyn > e_thr_unfz):       freeze_y=False
                if (not freeze_t) and (ethn < e_thr_freeze): freeze_t=True
                elif freeze_t and (ethn > e_thr_unfz):       freeze_t=False

            # --- Mapeo a ruedas (alpha = theta + pi/4) ---
            alpha_m = th + math.pi/4.0
            A = np.array([
                [ math.sqrt(2)*math.sin(alpha_m), -math.sqrt(2)*math.cos(alpha_m), -(L+l)],
                [ math.sqrt(2)*math.cos(alpha_m),  math.sqrt(2)*math.sin(alpha_m),  (L+l)],
                [ math.sqrt(2)*math.cos(alpha_m),  math.sqrt(2)*math.sin(alpha_m), -(L+l)],
                [ math.sqrt(2)*math.sin(alpha_m), -math.sqrt(2)*math.cos(alpha_m),  (L+l)]
            ], float)
            u_vec = np.array([ux, uy, uth], float)
            w_cmd = A @ u_vec
            bot.set_wheel_velocities(w_cmd)

            # --- Hamiltoniano canónico ---
            m_tot = m_base + 4*m_w + m_L
            Iz_chasis = (1/3)*m_base*(L**2 + l**2)
            Iz_rueda  = m_w*(L**2 + l**2) + 0.25*m_w*(r_w**2)
            Iz        = Iz_chasis + 4*Iz_rueda
            H_real = 0.5*m_tot*(ux**2 + uy**2) + 0.5*Iz*(uth**2)

            # =================== CAMBIO DE MASA ===================
            if (not mass_switched) and (t >= (t_switch if ML_SWITCH_T is not None else (S*0.5))):
                if bot.cuboid is not None:
                    try:
                        bot.sim.setShapeMass(bot.cuboid, ML_NEW)
                        print(f"[INFO] m_L cambió de {m_L:.6f} a {ML_NEW:.6f} kg @ t={t:.2f}s")
                        m_L = ML_NEW
                    except Exception as e:
                        print("[WARN] Falló setShapeMass; actualizo solo m_L interno. Detalle:", e)
                        m_L = ML_NEW
                else:
                    m_L = ML_NEW
                    print(f"[INFO] m_L interno cambió a {m_L:.6f} kg @ t={t:.2f}s (sin cuboid)")
                mass_switched = True
                switch_time_logged = t  # para marcar en gráficas
            # ======================================================

            # --- Predicción con TF (3 entradas) ---
            H_pred_inst = None
            if model is not None:
                X = np.array([[ux, uy, uth]], dtype=np.float32)
                try:
                    Hp = model.predict(X, verbose=0).ravel()[0]
                    H_pred_inst = float(Hp)
                except Exception as e:
                    if len(H_pred_list) == 0:
                        print("[WARN] Falló la predicción con TF en línea. Detalle:", e)
                    H_pred_inst = None

            # --- Disparador por discrepancia de H: UNFREEZE + HOLD + READAPTACIÓN ---
            trigger = False
            if H_pred_inst is not None and np.isfinite(H_pred_inst):
                diff_abs = abs(H_pred_inst - H_real)
                denom    = max(abs(H_real), H_EPS_DEN)
                diff_rel = diff_abs / denom
                if (diff_abs > H_DIFF_ABS_THR) or (diff_rel > H_DIFF_REL_THR):
                    print(diff_abs)
                    print(diff_rel)
                    trigger = True
                    freeze_x = freeze_y = freeze_t = False
                    hold_unfreeze_until = t + UNFREEZE_H_HOLD
                    h_unfreeze_events.append((t, diff_abs, diff_rel))
                    # Guardar en buffer
                    buffer.push(ux, uy, uth, H_real)

            # Siempre llenamos buffer para robustez
            if not trigger:
                buffer.push(ux, uy, uth, H_real)

            # Online updates con cooldown y si hay suficientes datos
            if model is not None and buffer.ready() and (t - last_trigger_time) >= COOLDOWN_BETWEEN_TRIGGERS:
                for _ in range(ONLINE_STEPS_PER_TRIGGER):
                    Xb, yb = buffer.sample(ONLINE_BATCH_SIZE)
                    model.train_on_batch(Xb, yb)
                last_trigger_time = t
                
                

            # --- Logs ---
            t_log.append(t); p_log.append(p); pd_log.append([xd,yd,thd]); w_log.append(w_cmd)
            ux_log.append(ux); uy_log.append(uy); uth_log.append(uth)
            ex_log.append(ex); ey_log.append(ey); eth_log.append(eth)
            exn_log.append(exn); eyn_log.append(eyn); ethn_log.append(ethn)
            Wx_log.append(w_x.copy()); Wy_log.append(w_y.copy()); Wt_log.append(w_t.copy())
            H_log.append(H_real); H_pred_list.append(H_pred_inst)

            t_prev=t; p_prev=p
            ex_prev,ey_prev,eth_prev = ex,ey,eth

            time.sleep(0.01)

    finally:
        bot.stop(); print("[INFO] Parado.")

    # -------- A arreglos --------
    t_log=np.array(t_log)
    p_log=np.array(p_log).T if p_log else np.zeros((3,0))
    pd_log=np.array(pd_log).T if pd_log else np.zeros((3,0))
    w_log=np.array(w_log).T if w_log else np.zeros((4,0))
    ux_log=np.array(ux_log); uy_log=np.array(uy_log); uth_log=np.array(uth_log)
    ex_log=np.array(ex_log); ey_log=np.array(ey_log); eth_log=np.array(eth_log)
    exn_log=np.array(exn_log); eyn_log=np.array(eyn_log); ethn_log=np.array(ethn_log)
    Wx_log=np.array(Wx_log).T if Wx_log else np.zeros((3,0))
    Wy_log=np.array(Wy_log).T if Wy_log else np.zeros((3,0))
    Wt_log=np.array(Wt_log).T if Wt_log else np.zeros((3,0))
    H_log=np.array(H_log, float)

    # Procesar H_pred
    if len(H_pred_list) == len(t_log):
        H_pred = np.array([np.nan if v is None else float(v) for v in H_pred_list], float)
    else:
        H_pred = None

    # ====== Métricas de H si hay predicción válida ======
    if H_pred is not None and np.isfinite(H_pred).any():
        mask = np.isfinite(H_pred) & np.isfinite(H_log)
        if mask.sum() > 5:
            err  = H_pred[mask] - H_log[mask]
            mse  = np.mean(err**2)
            rmse = np.sqrt(mse)
            mae  = np.mean(np.abs(err))
            r2   = r2_score_np(H_log[mask], H_pred[mask])
            rng  = (H_log[mask].max() - H_log[mask].min()) + 1e-12
            stdH = np.std(H_log[mask]) + 1e-12
            print(f"[Comparación H] RMSE={rmse:.6e}  MAE={mae:.6e}  R2={r2:.6f}  "
                  f"NRMSE(range)={rmse/rng:.6e}  NRMSE(std)={rmse/stdH:.6e}")
        else:
            print("[INFO] Muy pocos puntos válidos para métricas de H.")

    # ================== Gráficas ==================
    def mark_switch(ax):
        if PLOT_SWITCH_MARK and (ML_SWITCH_T is None or switch_time_logged is not None):
            ts = (switch_time_logged if switch_time_logged is not None else (t_log[-1]/2.0))
            ax.axvline(ts, linestyle='--', linewidth=1.2)
            ax.text(ts, ax.get_ylim()[1]*0.9, 'm_L cambiado', rotation=90, va='top', ha='right')

    plt.figure("Posición vs tiempo", figsize=(8,8))
    for i, lab in enumerate(['x [m]','y [m]', r'$\theta$ [rad]']):
        ax = plt.subplot(3,1,i+1); ax.grid(True)
        if pd_log.shape[1]>0: ax.plot(t_log, pd_log[i], '--', lw=1.5, label=f'{lab}_d')
        if p_log.shape[1]>0:  ax.plot(t_log, p_log[i], lw=1.5, label=lab)
        ax.set_ylabel(lab)
        if i==2: ax.set_xlabel('t [s]')
        ax.legend(loc='best'); mark_switch(ax);

    plt.figure("Trayectoria XY", figsize=(7,7))
    ax = plt.gca(); ax.grid(True); ax.set_aspect('equal', adjustable='box')
    if pd_log.shape[1]>0: ax.plot(pd_log[0], pd_log[1], 'k--', lw=1.5, label='ref')
    if p_log.shape[1]>0:
        ax.plot(p_log[0], p_log[1], 'b', lw=1.8, label='real')
        if p_log.shape[1]>0:
            draw_robot_ax(ax, p_log[:,-1], L, l)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.legend(loc='best'); ax.set_title(f"Seguimiento EKF-SNPID ({REF_MODO})")

    plt.figure("Control (u_x, u_y, u_theta)", figsize=(9,4))
    ax = plt.gca(); ax.grid(True)
    ax.plot(t_log, ux_log, label='u_x')
    ax.plot(t_log, uy_log, label='u_y')
    ax.plot(t_log, uth_log, label='u_θ')
    ax.set_xlabel('t [s]'); ax.set_ylabel('Acción de control'); ax.legend(loc='best'); mark_switch(ax); 

    plt.figure("Errores", figsize=(9,4))
    ax = plt.gca(); ax.grid(True)
    ax.plot(t_log, ex_log, label='e_x')
    ax.plot(t_log, ey_log, label='e_y')
    ax.plot(t_log, eth_log, label='e_θ')
    ax.set_xlabel('t [s]'); ax.legend(loc='best'); mark_switch(ax); 

    plt.figure("Errores normalizados", figsize=(9,4))
    ax = plt.gca(); ax.grid(True)
    ax.plot(t_log, exn_log, label='e_{x,n}')
    ax.plot(t_log, eyn_log, label='e_{y,n}')
    ax.plot(t_log, ethn_log, label='e_{θ,n}')
    ax.axhline(e_thr_freeze, linestyle='--', label='freeze')
    ax.axhline(e_thr_unfz,   linestyle='--', label='unfreeze')
    ax.set_xlabel('t [s]'); ax.legend(loc='best'); mark_switch(ax); 

    plt.figure("Ganancias aprendidas", figsize=(9,9))
    ax=plt.subplot(3,1,1); ax.grid(True)
    if Wx_log.shape[1]>0: ax.plot(t_log, Wx_log[0], t_log, Wx_log[1], t_log, Wx_log[2])
    ax.legend(['Kp_x','Kd_x','Ki_x']); ax.set_title('X'); mark_switch(ax); 
    ax=plt.subplot(3,1,2); ax.grid(True)
    if Wy_log.shape[1]>0: ax.plot(t_log, Wy_log[0], t_log, Wy_log[1], t_log, Wy_log[2])
    ax.legend(['Kp_y','Kd_y','Ki_y']); ax.set_title('Y'); mark_switch(ax); 
    ax=plt.subplot(3,1,3); ax.grid(True)
    if Wt_log.shape[1]>0: ax.plot(t_log, Wt_log[0], t_log, Wt_log[1], t_log, Wt_log[2])
    ax.legend([r'Kp_θ',r'Kd_θ',r'Ki_θ']); ax.set_title('θ'); ax.set_xlabel('t [s]'); mark_switch(ax);

    if w_log.shape[1]>0:
        plt.figure("Velocidades de rueda (comando)", figsize=(9,4))
        ax = plt.gca(); ax.grid(True)
        for i in range(4): ax.plot(t_log, w_log[i], label=f'v_{i+1}')
        ax.set_xlabel('t [s]'); ax.set_ylabel('rad/s'); ax.legend(loc='best'); mark_switch(ax);

    plt.figure("Hamiltoniano", figsize=(10,4))
    ax = plt.gca(); ax.grid(True)
    ax.plot(t_log, H_log, lw=1.2, label='H real (físico)')
    if H_pred is not None and np.isfinite(H_pred).any():
        ax.plot(t_log, H_pred, '--', label='H predicho (TF)')
    ax.set_xlabel('t [s]'); ax.set_ylabel('Energía [J]')
    ax.set_title('Hamiltoniano canónico')
    ax.legend(loc='best'); mark_switch(ax); 

    if H_pred is not None and np.isfinite(H_pred).any():
        plt.figure("Error del Hamiltoniano", figsize=(10,3))
        ax = plt.gca()
        diff = H_pred - H_log
        ax.plot(t_log, diff)
        ax.grid(True); ax.set_xlabel("t [s]"); ax.set_ylabel("H_pred - H_real [J]")
        ax.set_title("Error absoluto del Hamiltoniano"); mark_switch(ax); 
        plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
