% Implementación del controlador EKF-SNPID con anti-windup
% para un YouBot omnidireccional

clear; clc; close all;

%% Parámetros del robot (YouBot con 4 ruedas Mecanum)
L  = 0.2355;      % m, medio largo entre ejes delantero/trasero
l  = 0.15;        % m, medio ancho entre ruedas izquierda/derecha
r  = 0.0475;      % m, radio de rueda
dt = 0.05;        % s, tiempo de muestreo
Tsim = 130;        % s, tiempo total de simulación
N = round(Tsim/dt);

%% Trayectoria de referencia (elegir una: sinusoidal o rosa)
t = (0:N-1)*dt;
% Trayectoria sinusoidal
xref = 0.1 * t;
yref = 0.2 * sin(0.5*t);
thref= -pi/8 * ones(1,N);

% Trayectoria “rosa” con offset:
% a    = 0.2 + 0.05*cos(3*0.05*t);
% xref = a .* cos(0.05*t)-0.25;
% yref = a .* sin(0.05*t);
% thref= pi/4 * ones(1,N);

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
% w_x = 0.1 * randn(3,1); P_x = P_init;
% w_y = 0.1 * randn(3,1); P_y = P_init;
% w_t = 0.1 * randn(3,1); P_t = P_init;

%% Prealocar históricos
ux   = zeros(1,N); uy   = zeros(1,N); uth = zeros(1,N);
ex   = zeros(1,N); ey   = zeros(1,N); eth = zeros(1,N);
hist_wx = zeros(3,N);   % [Kp_x; Ki_x; Kd_x]
hist_wy = zeros(3,N);   % [Kp_y; Ki_y; Kd_y]
hist_wt = zeros(3,N);   % [Kp_th; Ki_th; Kd_th]
hist_wx(:,1) = w_x; 
hist_wy(:,1) = w_y;
hist_wt(:,1) = w_t;

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
    
    % Guardar historia de pesos
    hist_wx(:,k) = w_x;
    hist_wy(:,k) = w_y;
    hist_wt(:,k) = w_t;

    % 5) Actualizar cinematica del robot (modelo cinemático directo simplificado)
    state(:,k) = state(:,k-1) + dt * [ ux(k); uy(k); uth(k) ];
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

subplot(2,2,4)
plot(t, hist_wx(1,:), 'r', t, hist_wx(2,:), 'g', t, hist_wx(3,:), 'b');
hold on
plot(t, hist_wy(1,:), '--r', t, hist_wy(2,:), '--g', t, hist_wy(3,:), '--b');
plot(t, hist_wt(1,:), ':r', t, hist_wt(2,:), ':g', t, hist_wt(3,:), ':b');
grid on
xlabel('Tiempo [s]')
ylabel('Valor de ganancia')
title('Evolución de K_p, K_i, K_d')
legend('Kp_x','Ki_x','Kd_x', ...
       'Kp_y','Ki_y','Kd_y', ...
       'Kp_\theta','Ki_\theta','Kd_\theta', ...
       'Location','best')


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