% EKF_SNPID_Youbot_Freeze_PureNorm.m
% Congelamiento por pura normalización del error (sin ventanas).
% Histeresis simple: freeze si e_norm < thr_freeze, unfreeze si e_norm > thr_unfz.

clear; clc; close all;

%% Parámetros del robot (YouBot Mecanum)
L  = 0.1981;  l  = 0.199;  r  = 0.0475;
dt = 0.05;    Tsim = 280;
N  = round(Tsim/dt);

% Parámetros 
m_base = 16;          % kg, masa del chasis
m_w    = 2;           % kg por rueda
m_L    = 0;           % kg, carga centrada
a_box  = 0.30;   % m, largo de la caja (en planta) si aplica
b_box  = 0.25;   % m, ancho de la caja (en planta) si aplica

%% Referencia (sinusoidal) + pausa dura
t_original = (0:N-1)*dt;
xref_original  = 0.2 * t_original;
yref_original  = 0.4 * sin(0.2*t_original);
thref_original = -pi/8 * ones(1,N);

%Trayectoria “rosa” con offset:
% a    = 1 + 0.5*cos(5*0.05*t_original);
% xref_original = a .* cos(0.05*t_original)-1.5;
% yref_original = a .* sin(0.05*t_original);
% thref_original= pi/4 * ones(1,N);

pause_start    = 35;   % s
pause_duration = 0;   % s
idx_start = round(pause_start/dt);
idx_pause = round(pause_duration/dt);
idx_end   = idx_start + idx_pause;

x_hold  = xref_original(idx_start);
y_hold  = yref_original(idx_start);
th_hold = thref_original(idx_start);

xref = [xref_original(1:idx_start), repmat(x_hold,1,idx_pause), xref_original(idx_start+1:end)];
yref = [yref_original(1:idx_start), repmat(y_hold,1,idx_pause), yref_original(idx_start+1:end)];
thref= [thref_original(1:idx_start), repmat(th_hold,1,idx_pause), thref_original(idx_start+1:end)];

t = (0:length(xref)-1)*dt;
N = length(t);

%% EKF-SNPID
P_init = eye(3);
Q      = 0.1 * eye(3);
R      = 1e-4;

n_x  = [0.1; 0.1; 0.01];
n_y  = [0.1; 0.1; 0.01];
n_th = [0.1; 0.1; 0.01];

alpha = 1.5;
uMax  = [1.5; 1.5; 1.5];

%% Pura normalización (sin ventanas)
e_norm_thr_freeze = 0.05;   % congela si e_norm < 5%
e_norm_thr_unfz   = 0.1;   % descongela si e_norm > 15%  (debe ser > freeze)

%% Estados e históricos
state = zeros(3,N); state(:,1) = [0;0;0];
ux = zeros(1,N); uy = zeros(1,N); uth = zeros(1,N);
ex = zeros(1,N); ey = zeros(1,N); eth = zeros(1,N);
W_x = zeros(3,N); W_y = zeros(3,N); W_t = zeros(3,N);
pp = zeros(1,N); p_step = 0;

% Normalizados (para plots/depuración)
exn = zeros(1,N); eyn = zeros(1,N); ethn = zeros(1,N);

% Inicialización de pesos
w_x = [0; 0; 0];
w_y = [0; 0; 0];
w_t = [0; 0; 0];

% w_x = [0.44; 0.16; 0.13];
% w_y = [3.9; 0.1; 0.4];
% w_t = [9.4; -.9; 1.1];

% w_x = [1.228; -0.06; 0.031];
% w_y = [0.8; -.04; .035];
% w_t = [1.54; -.14; .112];

P_x = P_init; W_x(:,1) = w_x;
P_y = P_init; W_y(:,1) = w_y;
P_t = P_init; W_t(:,1) = w_t;

% Integrales discretas (protegidas)
dx3 = 0; dy3 = 0; dt3 = 0;

% Flags de congelamiento por eje
freeze_x=false; freeze_y=false; freeze_t=false;

%% Bucle
for k = 2:N
    xk = state(1,k-1); yk = state(2,k-1); th = state(3,k-1);

    ex(k)  = xref(k) - xk;
    ey(k)  = yref(k) - yk;
    eth(k) = wrapToPi_local(thref(k) - th);

    % Componentes P, D, I
    dx1 = ex(k);               dx2 = ex(k) - ex(k-1);
    dy1 = ey(k);               dy2 = ey(k) - ey(k-1);
    dt1 = eth(k);              dt2 = eth(k) - eth(k-1);

    % Integrales: evita acumular si no hay avance (para k>=3)
    if k>=3
        p_step = norm(state(:,k-1) - state(:,k-2));
        if p_step > 1e-12
            dx3 = dx3 + ex(k);
            dy3 = dy3 + ey(k);
            dt3 = dt3 + eth(k);
        end
    else
        dx3 = dx3 + ex(k);
        dy3 = dy3 + ey(k);
        dt3 = dt3 + eth(k);
    end

    % Control + EKF (con freeze por eje)
    [ux(k), w_x, P_x] = ekf_snpid_freeze(dx1,dx2,dx3, w_x, P_x, Q, R, n_x, alpha, uMax(1), freeze_x);
    [uy(k), w_y, P_y] = ekf_snpid_freeze(dy1,dy2,dy3, w_y, P_y, Q, R, n_y, alpha, uMax(2), freeze_y);
    [uth(k),w_t, P_t] = ekf_snpid_freeze(dt1,dt2,dt3, w_t, P_t, Q, R, n_th,alpha, uMax(3), freeze_t);

    % Dinámica con bloqueo en la pausa
    if k >= idx_start && k < idx_end
    % if k > 305 && k < 1000
        state(:,k) = state(:,k-1);
        m_L    = 1;
    else
        state(:,k) = state(:,k-1) + dt * [ux(k); uy(k); uth(k)];
    end

    % Log de pesos
    W_x(:,k) = w_x; W_y(:,k) = w_y; W_t(:,k) = w_t;
    pp(:, k) = p_step;

    % ===== Congelamiento (histeresis simple) =====
    exn(k)  = abs(ex(k));
    eyn(k)  = abs(ey(k));
    ethn(k) = abs(eth(k));

    % X
    if ~freeze_x && (exn(k) < e_norm_thr_freeze)
        freeze_x = true;
    elseif freeze_x && (exn(k) > e_norm_thr_unfz)
        freeze_x = false;
    end

    % Y
    if ~freeze_y && (eyn(k) < e_norm_thr_freeze)
        freeze_y = true;
    elseif freeze_y && (eyn(k) > e_norm_thr_unfz)
        freeze_y = false;
    end

    % THETA
    if ~freeze_t && (ethn(k) < e_norm_thr_freeze)
        freeze_t = true;
    elseif freeze_t && (ethn(k) > e_norm_thr_unfz)
        freeze_t = false;
    end
end

%% ===== Hamiltoniano (m_base=16 kg, m_rueda=2 kg c/u) =====

m_tot = m_base + 4*m_w + m_L;

Iz_chasis = (1/3) * m_base * (L^2 + l^2);                     % placa de 2L x 2l
Iz_rueda  = m_w * (L^2 + l^2) + 0.25 * m_w * r^2;             % por rueda
Iz        = Iz_chasis + 4 * Iz_rueda;

vx = ux; vy = uy; w = uth;                                    % en este modelo, u ≈ velocidades
H  = 0.5 * m_tot .* (vx.^2 + vy.^2) + 0.5 * Iz .* (w.^2);     % J

%% ===== Hamiltoniano con carga centrada (paramétrico) =====
% include_wheel_spin   = true;   % energía de giro de ruedas (opcional)
% use_load_box_inertia = false;   % true: la carga tiene inercia propia; false: punto centrado
% 
% m_tot = m_base + 4*m_w + m_L;
% 
% % Inercia alrededor de z (sin doble contabilidad del giro de ruedas)
% Izz_chasis_CoM = (1/3) * m_base * (L^2 + l^2);
% Izz_wheels_loc = 4 * m_w * (L^2 + l^2);  % ruedas como masas en las esquinas (±L,±l)
% 
% if use_load_box_inertia
%     Izz_load_CoM = (1/12) * m_L * (a_box^2 + b_box^2);  % caja delgada centrada
% else
%     Izz_load_CoM = 0;  % carga como punto centrado
% end
% 
% Iz_tot = Izz_chasis_CoM + Izz_wheels_loc + Izz_load_CoM;
% 
% % Velocidades del cuerpo (en tu modelo u ≈ twist del cuerpo)
% vx = ux; vy = uy; w = uth;
% 
% % Energía del cuerpo (traslación + rotación)
% H_body = 0.5 * m_tot .* (vx.^2 + vy.^2) + 0.5 * Iz_tot .* (w.^2);
% 
% % Energía de giro de ruedas (opcional, usando inversa mecanum típica)
% if include_wheel_spin
%     a = (L + l);
%     w1 = (1/r) * (vx - vy - a .* w);
%     w2 = (1/r) * (vx + vy + a .* w);
%     w3 = (1/r) * (vx + vy - a .* w);
%     w4 = (1/r) * (vx - vy + a .* w);
%     Iw = 0.5 * m_w * r^2;    % disco sólido aprox. sobre su propio eje
%     H_spin = 0.5 * Iw .* (w1.^2 + w2.^2 + w3.^2 + w4.^2);
% else
%     H_spin = zeros(size(H_body));
% end
% 
% % Hamiltoniano total
% H = H_body + H_spin;


%% Plots
figure;
plot(state(1,:), state(2,:), 'b', xref, yref, '--r'); grid on;
xlabel('x [m]'); ylabel('y [m]'); legend('Real','Ref','Location','best'); title('Plano XY');

figure;
plot(t, ex, t, ey, t, eth); grid on; legend('e_x','e_y','e_\theta'); xlabel('t [s]'); title('Errores');

figure;
plot(t, ux, t, uy, t, uth); grid on; legend('u_x','u_y','u_\theta'); xlabel('t [s]'); title('Control');

figure;
plot(t, exn, t, eyn, t, ethn); grid on; hold on;
yline(e_norm_thr_freeze,'--k','freeze'); yline(e_norm_thr_unfz,'--r','unfreeze');
legend('e_{x,n}','e_{y,n}','e_{\theta,n}','Location','best');
xlabel('t [s]'); title('Errores normalizados (con histéresis)');

figure('Name','Ganancias','NumberTitle','off');
subplot(3,1,1); plot(t,W_x(1,:), t,W_x(2,:), t,W_x(3,:)); grid on; legend('Kp_x','Ki_x','Kd_x'); title('X');
subplot(3,1,2); plot(t,W_y(1,:), t,W_y(2,:), t,W_y(3,:)); grid on; legend('Kp_y','Ki_y','Kd_y'); title('Y');
subplot(3,1,3); plot(t,W_t(1,:), t,W_t(2,:), t,W_t(3,:)); grid on; legend('Kp_\theta','Ki_\theta','Kd_\theta'); title('\theta');

figure('Name','Hamiltoniano','NumberTitle','off');
plot(t, H, 'LineWidth',1.2); grid on;
xlabel('t [s]'); ylabel('Energía [J]');
title('Hamiltoniano (m_{base}=16 kg, m_{rueda}=2 kg c/u)');

figure();
plot(t,pp)
title('Cambio de pose');

%% ===== Funciones =====
function [u_sat, w_new, P_new] = ekf_snpid_freeze(x1,x2,x3, w, P, Q, R, n, alpha, umax, freeze_flag)
    x = [x1; x2; x3];
    v = w' * x;
    u = alpha * tanh(v);

    % Anti-windup por back-calculation simple
    u_sat = max(-umax, min(umax, u));
    es    = u_sat - u;
    e_aw  = x1 + es;  % innovación simple

    % Jacobiano y covarianza de innovación
    sech2 = (1 - tanh(v)^2);
    H = alpha * sech2 * x;
    S = H' * P * H + R;

    if freeze_flag
        w_new  = w; P_new = P; return;
    end

    % EKF update
    K     = (P * H) / S;
    P_new = P - K * H' * P + Q;
    w_new = w + (n .* K) * e_aw;
end

function ang = wrapToPi_local(ang)
    ang = mod(ang + pi, 2*pi) - pi;
end
