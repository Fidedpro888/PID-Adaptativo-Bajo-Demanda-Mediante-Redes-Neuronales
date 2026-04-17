% EKF_SNPID_Youbot.m
% Implementación del controlador EKF-SNPID con anti-windup
% para un YouBot omnidireccional, usando los parámetros del paper

clear; clc; close all;

%% Parámetros del robot (YouBot con 4 ruedas Mecanum)
L  = 0.2355;      % m, medio largo entre ejes delantero/trasero
l  = 0.15;        % m, medio ancho entre ruedas izquierda/derecha
r  = 0.0475;      % m, radio de rueda
dt = 0.05;        % s, tiempo de muestreo
Tsim = 80;        % s, tiempo total de simulación
N = round(Tsim/dt);


%% Trayectoria de referencia (elegir una: sinusoidal o rosa)
t_original = (0:N-1)*dt;
% xref_original  = 0.2 * t_original;
% yref_original  = 0.4 * sin(0.2*t_original);
% thref_original = -pi/8 * ones(1,N);

% Trayectoria “rosa” con offset:
a    = 1 + 0.5*cos(5*0.05*t_original);
xref_original = a .* cos(0.05*t_original)-1.5;
yref_original = a .* sin(0.05*t_original);
thref_original= pi/4 * ones(1,N);
pause_start = 15;   % segundos
pause_duration = 20; % segundos

idx_start = round(pause_start/dt);
idx_pause = round(pause_duration/dt);

% Valores en el instante de pausa
x_hold  = xref_original(idx_start);
y_hold  = yref_original(idx_start);
th_hold = thref_original(idx_start);

% Construir trayectoria extendida
xref = [xref_original(1:idx_start), repmat(x_hold,1,idx_pause), xref_original(idx_start+1:end)];
yref = [yref_original(1:idx_start), repmat(y_hold,1,idx_pause), yref_original(idx_start+1:end)];
thref= [thref_original(1:idx_start), repmat(th_hold,1,idx_pause), thref_original(idx_start+1:end)];

% Nuevo tiempo total
t = (0:length(xref)-1)*dt;
N = length(t);
p = zeros(1,N);

%% Parámetros del controlador EKF-SNPID
% Covarianzas EKF
P_init = eye(3);
Q      = 0.1 * eye(3);
R      = 1e-4;

% Ganancias de aprendizaje por peso (vector n)
n_x = [0.1; 0.1; 0.01];
n_y = [0.1; 0.1; 0.01];
n_th= [0.1; 0.1; 0.01];

alpha = 0.3;        % factor de escala de tanh

% Límites de saturación (misma en x, y, theta)
uMax = [ 0.3; 0.3; 0.3];  % [m/s; m/s; rad/s]

%% Inicialización de estados y EKF para cada DOF
state = zeros(3, N);     % [x; y; theta]
state(:,1) = [0;0;0];

% Pesos del neurona PID (w1→Kp, w2→Ki, w3→Kd) y matrices EKF
w_x = zeros(3,1); P_x = P_init;
w_y = zeros(3,1); P_y = P_init;
w_t = zeros(3,1); P_t = P_init;

%% Prealocar históricos
ux   = zeros(1,N); uy   = zeros(1,N); uth = zeros(1,N);
ex   = zeros(1,N); ey   = zeros(1,N); eth = zeros(1,N);
W_x = zeros(3,N);  % [Kp_x; Ki_x; Kd_x]
W_y = zeros(3,N);  % [Kp_y; Ki_y; Kd_y]
W_t = zeros(3,N);  % [Kp_th; Ki_th; Kd_th]

%% Bucle de simulación / control
for k = 2:N
    % 1) Medición de estado actual
    xk = state(1,k-1);
    yk = state(2,k-1);
    th = state(3,k-1);
    
    % 2) Error de seguimiento
    ex(k)  = xref(k) - xk;
    ey(k)  = yref(k) - yk;
    eth(k) = wrapToPi(thref(k) - th);
    
    % 3) Entradas P, D, I (diferen y suma discreta)
    % --- Para x
    dx1 = ex(k);
    dx2 = ex(k) - ex(k-1);
    dx3 = sum(ex(1:k));
    % --- Para y
    dy1 = ey(k);
    dy2 = ey(k) - ey(k-1);
    dy3 = sum(ey(1:k));
    % --- Para theta
    dt1 = eth(k);
    dt2 = eth(k) - eth(k-1);
    dt3 = sum(eth(1:k));
    
    % 4) Computar controlador neuronal con EKF y anti-windup
    [ux(k), w_x, P_x] = ekf_snpid(dx1,dx2,dx3, w_x, P_x, Q, R, n_x, alpha, uMax(1));
    [uy(k), w_y, P_y] = ekf_snpid(dy1,dy2,dy3, w_y, P_y, Q, R, n_y, alpha, uMax(2));
    [uth(k),w_t, P_t] = ekf_snpid(dt1,dt2,dt3, w_t, P_t, Q, R, n_th, alpha, uMax(3));
    
    idx_end = idx_start + idx_pause;

    if k >= idx_start && k < idx_end
        state(:,k) = state(:,k-1); % robot bloqueado
    else
        state(:,k) = state(:,k-1) + dt * [ ux(k); uy(k); uth(k) ];
    end

    W_x(:,k) = w_x;
    W_y(:,k) = w_y;
    W_t(:,k) = w_t;

end

%% Plots de resultados
figure;
subplot(2,2,1)
plot(state(1,:), state(2,:), 'b', xref, yref, '--r'); grid on;
xlabel('x [m]'); ylabel('y [m]');
legend('Trayectoria real','Referencia','Location','best')
title('Plano XY')

subplot(2,2,2)
plot(t, ex, t, ey, t, eth);
legend('e_x','e_y','e_\theta'); grid on;
xlabel('Tiempo [s]'); title('Errores de seguimiento')

subplot(2,2,3)
plot(t, ux, t, uy, t, uth); grid on;
legend('u_x','u_y','u_\theta');
xlabel('Tiempo [s]'); title('Señales de control')

figure('Name','Evolución de Ganancias Neuronales','NumberTitle','off');

subplot(3,1,1)
plot(t, W_x(1,:), 'r', t, W_x(2,:), 'g', t, W_x(3,:), 'b');
legend('Kp_x','Ki_x','Kd_x'); grid on;
xlabel('Tiempo [s]'); ylabel('Ganancia'); title('Ganancias en eje X');

subplot(3,1,2)
plot(t, W_y(1,:), 'r', t, W_y(2,:), 'g', t, W_y(3,:), 'b');
legend('Kp_y','Ki_y','Kd_y'); grid on;
xlabel('Tiempo [s]'); ylabel('Ganancia'); title('Ganancias en eje Y');

subplot(3,1,3)
plot(t, W_t(1,:), 'r', t, W_t(2,:), 'g', t, W_t(3,:), 'b');
legend('Kp_\theta','Ki_\theta','Kd_\theta'); grid on;
xlabel('Tiempo [s]'); ylabel('Ganancia'); title('Ganancias en orientación \theta');


function [u_sat, w_new, P_new] = ekf_snpid(x1,x2,x3, w, P, Q, R, n, alpha, umax)
% ekf_snpid: controlador single-neuron PID entrenado online con EKF + anti-windup
% Entradas:
%   x1,x2,x3 : entradas P, D y I del error
%   w, P     : pesos actuales y covarianza EKF (3×1 y 3×3)
%   Q, R     : covarianzas de proceso y medición
%   n        : vector de ganancias de aprendizaje [n1;n2;n3]
%   alpha    : factor de escala para tanh
%   umax     : saturación unidimensional
%
% Salidas:
%   u_sat    : señal de control saturada
%   w_new,P_new : nuevos pesos y covarianza EKF

    % 1) Calcular sumatoria ponderada v y salida u (antes de saturar)
    x = [x1; x2; x3];
    v = w' * x;
    u = alpha * tanh(v);
    
    % 2) Anti-windup back-calculation
    %    saturamos u y generamos e_aw
    u_sat = max(-umax, min(umax, u));
    es    = u_sat - u;       % error de saturación
    if abs(es)>0
        e_aw = x1 + es;      % solo modifica parte I internamente
    else
        e_aw = x1;
    end
    
    % 3) EKF: actualización de pesos
    %    a) Jacobiano H = ∂u/∂w  (3×1)
    sech2 = (1 - tanh(v)^2);
    H = alpha * sech2 * x;   % H = [∂u/∂w1; ...; ∂u/∂w3]
    
    %    b) Ganancia de Kalman (3×1)
    %       K = P*H / (H'*P*H + R)
    S = H' * P * H + R;
    K = (P * H) / S;
    
    %    c) Actualizar covarianza
    P_new = P - K * H' * P + Q;
    
    % 4) Ajuste final: escalamos la ganancia de Kalman
    %    (en el paper proponen n .* K; para más flexibilidad)
    w_new = w + (n .* K) * e_aw;
end