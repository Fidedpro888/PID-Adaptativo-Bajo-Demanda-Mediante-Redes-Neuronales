% Control PID discreto clásico para YouBot omnidireccional
% usando los mismos Kp, Ki, Kd que reporta el paper

clear; clc; close all;

%% Parámetros del robot (YouBot Mecanum 4 ruedas)
L   = 0.2355;    % m
l   = 0.15;      % m
r   = 0.0475;    % m
dt  = 0.05;      % s de muestreo
Tsim= 130;        % s de simulación total
N   = round(Tsim/dt);

%% Trayectoria de referencia
t_original = (0:N-1)*dt;

%Trayectoria sinusoidal
xref_original  = 0.1 * t_original;
yref_original  = 0.2 * sin(0.5*t_original);
thref_original = -pi/8 * ones(1,N);

% Trayectoria “rosa” con offset:
% a    = 0.2 + 0.05*cos(10*0.05*t_original);
% xref_original = a .* cos(0.05*t_original)-0.25;
% yref_original = a .* sin(0.05*t_original);
% thref_original= pi/4 * ones(1,N);

pause_start = 15;   % segundos
pause_duration = 50; % segundos

idx_start = round(pause_start/dt);
idx_pause = round(pause_duration/dt);
idx_end = idx_start + idx_pause;

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


%% Ganancias PID según el paper
Kp = [0.3; 0.3; 0.2];   % [Kp_x; Kp_y; Kp_th]
Ki = [0.2; 0.2; 0.1];   % [Ki_x; Ki_y; Ki_th]
Kd = [0.1; 0.1; 0.05];  % [Kd_x; Kd_y; Kd_th]

%% Inicialización de estados y errores
state = zeros(3, N);    % [x; y; theta]
state(:,1) = [0;0;0];

ex   = zeros(1,N); ey   = zeros(1,N); eth = zeros(1,N);
sum_ex = 0; sum_ey = 0; sum_eth = 0;

ux = zeros(1,N); uy = zeros(1,N); uth = zeros(1,N); p = zeros(1,N);

%% Bucle principal de control + simulación
for k = 2:N
    % --- Medición de pose actual
    xk = state(1,k-1);
    yk = state(2,k-1);
    th = state(3,k-1);
    
    % --- Cálculo de errores
    ex(k)  = xref(k)  - xk;
    ey(k)  = yref(k)  - yk;
    eth(k) = wrapToPi(thref(k) - th);
    
    % --- Integral de errores
    p(k) = norm(state(:,k) - state(:,k-1));
    % if abs(p(k) - p(k-1)) < 1e-6
    % 
    % else
    %     sum_ex   = sum_ex   + ex(k);
    %     sum_ey   = sum_ey   + ey(k);
    %     sum_eth  = sum_eth  + eth(k);
    % end

    sum_ex   = sum_ex   + ex(k);
    sum_ey   = sum_ey   + ey(k);
    sum_eth  = sum_eth  + eth(k);

    % --- Derivada de errores
    de_x   = ex(k)   - ex(k-1);
    de_y   = ey(k)   - ey(k-1);
    de_th  = eth(k)  - eth(k-1);
    
    % --- Ley de control PID discreto
    ux(k)  = Kp(1)*ex(k)  + Ki(1)*sum_ex   + Kd(1)*de_x;
    uy(k)  = Kp(2)*ey(k)  + Ki(2)*sum_ey   + Kd(2)*de_y;
    uth(k) = Kp(3)*eth(k) + Ki(3)*sum_eth  + Kd(3)*de_th;
    
        % --- Anti-windup: saturación de control y corrección de integral
    sat_limit = 0.3;

    % Saturación y corrección ux
    if abs(ux(k)) > sat_limit
        ux(k) = sign(ux(k)) * sat_limit;
        sum_ex = sum_ex - ex(k);  % anti-windup
    end

    % Saturación y corrección uy
    if abs(uy(k)) > sat_limit
        uy(k) = sign(uy(k)) * sat_limit;
        sum_ey = sum_ey - ey(k);  % anti-windup
    end

    % Saturación y corrección uth
    if abs(uth(k)) > sat_limit
        uth(k) = sign(uth(k)) * sat_limit;
        sum_eth = sum_eth - eth(k);  % anti-windup
    end

    if k >= idx_start && k < idx_end
        state(:,k) = state(:,k-1); % robot bloqueado
    else
        state(:,k) = state(:,k-1) + dt * [ ux(k); uy(k); uth(k) ];
    end
end

%% Visualización de resultados
figure('Name','PID Convencional en YouBot','NumberTitle','off');

subplot(2,2,1)
plot(state(1,:), state(2,:), 'b', xref, yref, '--r');
grid on;
xlabel('x [m]'); ylabel('y [m]');
legend('trayectoria real','referencia','Location','best')
title('Trayectoria en el plano XY')

subplot(2,2,2)
plot(t, ex, t, ey, t, eth,'LineWidth',1);
grid on;
legend('e_x','e_y','e_\theta','Location','best')
xlabel('tiempo [s]'); title('Errores de seguimiento')

subplot(2,2,3)
plot(t, ux, t, uy, t, uth,'LineWidth',1);
grid on;
legend('u_x','u_y','u_\theta','Location','best')
xlabel('tiempo [s]'); title('Señales de control PID')

subplot(2,2,4)
plot(t,p)
