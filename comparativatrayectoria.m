% COMPARACION 3 CONTROLADORES EN YOUBOT (TRAYECTORIA ROSA)
% PID convencional vs EKF-SNPID (básico) vs Propuesto (EKF-SNPID + freeze)
% - Misma referencia "rosa"
% - Misma saturación (+/- 0.3)
% - Figuras: Seguimiento (1), Errores (1 con subplots), Control (1 con subplots)
% - Métricas impresas en tabla

clear; clc; close all;

%% Parámetros generales del robot y simulación
L   = 0.2355;    % m
l   = 0.15;      % m
r   = 0.0475;      % m
dt  = 0.05;        % s
Tsim= 130;         % s
N   = round(Tsim/dt);
t   = (0:N-1)*dt;

sat_limit = 0.3;   % saturación común para los 3 controladores

%% Referencia: TRAYECTORIA "ROSA" (única para los 3)
a    = 0.2 + 0.05*cos(3*0.05*t);
xref = a .* cos(0.05*t) - 0.25;
yref = a .* sin(0.05*t);
thref= pi/4 * ones(1,N);

%% --------- CONTROLADOR 1: PID CONVENCIONAL ---------
Kp = [0.3; 0.3; 0.2];   % [Kp_x; Kp_y; Kp_th]
Ki = [0.2; 0.2; 0.1];   % [Ki_x; Ki_y; Ki_th]
Kd = [0.1; 0.1; 0.05];  % [Kd_x; Kd_y; Kd_th]

state_PID = zeros(3,N); state_PID(:,1) = [0;0;0];
ex_PID=zeros(1,N); ey_PID=zeros(1,N); eth_PID=zeros(1,N);
ux_PID=zeros(1,N); uy_PID=zeros(1,N); uth_PID=zeros(1,N);
sum_ex=0; sum_ey=0; sum_eth=0;

for k = 2:N
    xk = state_PID(1,k-1); yk = state_PID(2,k-1); th = state_PID(3,k-1);
    ex_PID(k)  = xref(k) - xk;
    ey_PID(k)  = yref(k) - yk;
    eth_PID(k) = wrapToPi_local(thref(k) - th);

    de_x  = ex_PID(k)  - ex_PID(k-1);
    de_y  = ey_PID(k)  - ey_PID(k-1);
    de_th = eth_PID(k) - eth_PID(k-1);

    sum_ex  = sum_ex  + ex_PID(k);
    sum_ey  = sum_ey  + ey_PID(k);
    sum_eth = sum_eth + eth_PID(k);

    ux_PID(k)  = Kp(1)*ex_PID(k)  + Ki(1)*sum_ex  + Kd(1)*de_x;
    uy_PID(k)  = Kp(2)*ey_PID(k)  + Ki(2)*sum_ey  + Kd(2)*de_y;
    uth_PID(k) = Kp(3)*eth_PID(k) + Ki(3)*sum_eth + Kd(3)*de_th;

    % Saturación + anti-windup por back-calc simple
    [ux_PID(k), sum_ex]  = sat_aw(ux_PID(k),  sum_ex,  ex_PID(k),  sat_limit);
    [uy_PID(k), sum_ey]  = sat_aw(uy_PID(k),  sum_ey,  ey_PID(k),  sat_limit);
    [uth_PID(k),sum_eth] = sat_aw(uth_PID(k), sum_eth, eth_PID(k), sat_limit);

    state_PID(:,k) = state_PID(:,k-1) + dt * [ux_PID(k); uy_PID(k); uth_PID(k)];
end

%% --------- CONTROLADOR 2: EKF-SNPID (BÁSICO) ---------
% Pesos iniciales solicitados
w_x = [0.3; 0.2; 0.2];
w_y = [0.3; 0.2; 0.1];
w_t = [0.2; 0.1; 0.05];

P_init = eye(3);
Q      = 0.1*eye(3);
R      = 1e-4;
n_x = [0.1; 0.1; 0.01];
n_y = [0.1; 0.1; 0.01];
n_th= [0.1; 0.1; 0.01];
alpha_ekf = 1.5;

state_EKF = zeros(3,N); state_EKF(:,1) = [0;0;0];
ex_EKF=zeros(1,N); ey_EKF=zeros(1,N); eth_EKF=zeros(1,N);
ux_EKF=zeros(1,N); uy_EKF=zeros(1,N); uth_EKF=zeros(1,N);

P_x = P_init; P_y = P_init; P_t = P_init;
dx3=0; dy3=0; dt3=0;

for k = 2:N
    xk = state_EKF(1,k-1); yk = state_EKF(2,k-1); th = state_EKF(3,k-1);
    ex_EKF(k)  = xref(k) - xk;
    ey_EKF(k)  = yref(k) - yk;
    eth_EKF(k) = wrapToPi_local(thref(k) - th);

    dx1 = ex_EKF(k);  dx2 = ex_EKF(k)  - ex_EKF(k-1);  dx3 = dx3 + ex_EKF(k);
    dy1 = ey_EKF(k);  dy2 = ey_EKF(k)  - ey_EKF(k-1);  dy3 = dy3 + ey_EKF(k);
    dt1 = eth_EKF(k); dt2 = eth_EKF(k) - eth_EKF(k-1); dt3 = dt3 + eth_EKF(k);

    [ux_EKF(k), w_x, P_x]  = ekf_snpid(dx1,dx2,dx3, w_x, P_x, Q, R, n_x,  alpha_ekf, sat_limit);
    [uy_EKF(k), w_y, P_y]  = ekf_snpid(dy1,dy2,dy3, w_y, P_y, Q, R, n_y,  alpha_ekf, sat_limit);
    [uth_EKF(k),w_t, P_t]  = ekf_snpid(dt1,dt2,dt3, w_t, P_t, Q, R, n_th, alpha_ekf, sat_limit);

    state_EKF(:,k) = state_EKF(:,k-1) + dt * [ux_EKF(k); uy_EKF(k); uth_EKF(k)];
end

%% --------- CONTROLADOR 3: PROPUESTO (EKF-SNPID + FREEZE/HISTERESIS) ---------
% Pesos iniciales solicitados (otra copia para independencia)
w_xp = [0.3; 0.2; 0.2];
w_yp = [0.3; 0.2; 0.1];
w_tp = [0.2; 0.1; 0.05];

P_xp = P_init; P_yp = P_init; P_tp = P_init;
alpha_prop = 1.5; 
e_thr_freeze = 0.01;   % congela si |e| < 0.05 (m o rad)
e_thr_unfz   = 0.03;   % descongela si |e| > 0.15

state_PRO = zeros(3,N); state_PRO(:,1) = [0;0;0];
ex_PRO=zeros(1,N); ey_PRO=zeros(1,N); eth_PRO=zeros(1,N);
ux_PRO=zeros(1,N); uy_PRO=zeros(1,N); uth_PRO=zeros(1,N);

dx3p=0; dy3p=0; dt3p=0;
freeze_x=false; freeze_y=false; freeze_t=false;

for k = 2:N
    xk = state_PRO(1,k-1); yk = state_PRO(2,k-1); th = state_PRO(3,k-1);
    ex_PRO(k)  = xref(k) - xk;
    ey_PRO(k)  = yref(k) - yk;
    eth_PRO(k) = wrapToPi_local(thref(k) - th);

    % Derivadas e integrales (con gating por movimiento: opcional simple)
    dx1 = ex_PRO(k);  dx2 = ex_PRO(k)  - ex_PRO(k-1);
    dy1 = ey_PRO(k);  dy2 = ey_PRO(k)  - ey_PRO(k-1);
    dt1 = eth_PRO(k); dt2 = eth_PRO(k) - eth_PRO(k-1);

    if k>=3
        p_step = norm(state_PRO(:,k-1) - state_PRO(:,k-2));
        if p_step > 1e-12
            dx3p = dx3p + ex_PRO(k);
            dy3p = dy3p + ey_PRO(k);
            dt3p = dt3p + eth_PRO(k);
        end
    else
        dx3p = dx3p + ex_PRO(k);
        dy3p = dy3p + ey_PRO(k);
        dt3p = dt3p + eth_PRO(k);
    end

    % EKF-SNPID con bandera de freeze por eje
    [ux_PRO(k), w_xp, P_xp] = ekf_snpid_freeze(dx1,dx2,dx3p, w_xp, P_xp, Q, R, n_x,  alpha_prop, sat_limit, freeze_x);
    [uy_PRO(k), w_yp, P_yp] = ekf_snpid_freeze(dy1,dy2,dy3p, w_yp, P_yp, Q, R, n_y,  alpha_prop, sat_limit, freeze_y);
    [uth_PRO(k),w_tp,P_tp]  = ekf_snpid_freeze(dt1,dt2,dt3p, w_tp, P_tp, Q, R, n_th, alpha_prop, sat_limit, freeze_t);

    % Histeresis de congelamiento (con errores absolutos)
    if ~freeze_x && abs(ex_PRO(k)) < e_thr_freeze, freeze_x = true; end
    if  freeze_x && abs(ex_PRO(k)) > e_thr_unfz,   freeze_x = false; end

    if ~freeze_y && abs(ey_PRO(k)) < e_thr_freeze, freeze_y = true; end
    if  freeze_y && abs(ey_PRO(k)) > e_thr_unfz,   freeze_y = false; end

    if ~freeze_t && abs(eth_PRO(k)) < e_thr_freeze, freeze_t = true; end
    if  freeze_t && abs(eth_PRO(k)) > e_thr_unfz,   freeze_t = false; end

    state_PRO(:,k) = state_PRO(:,k-1) + dt * [ux_PRO(k); uy_PRO(k); uth_PRO(k)];
end

%% ----------------------- FIGURA 1: SEGUIMIENTO (3 pilares) -----------------------
fig1 = figure('Name','Seguimiento - 3 pilares','NumberTitle','off');
tl1 = tiledlayout(fig1,1,3,'TileSpacing','compact','Padding','compact');
title(tl1,'Seguimiento XY (trayectoria rosa con pausa)');

% Límites comunes para XY
x_all = [xref, state_PID(1,:), state_EKF(1,:), state_PRO(1,:)];
y_all = [yref, state_PID(2,:), state_EKF(2,:), state_PRO(2,:)];
x_rng = max(x_all)-min(x_all); y_rng = max(y_all)-min(y_all);
xm = 0.05*max(x_rng, eps); ym = 0.05*max(y_rng, eps);
xlimv = [min(x_all)-xm, max(x_all)+xm];
ylimv = [min(y_all)-ym, max(y_all)+ym];

% Col 1: PID
nexttile;
plot(xref, yref, 'k--','LineWidth',1.2); hold on; grid on;
plot(state_PID(1,:), state_PID(2,:), 'LineWidth',1.1);
axis equal; xlim(xlimv); ylim(ylimv);
xlabel('x [m]'); ylabel('y [m]'); title('PID');
legend('Referencia','Trayectoria','Location','best');

% Col 2: EKF-SNPID
nexttile;
plot(xref, yref, 'k--','LineWidth',1.2); hold on; grid on;
plot(state_EKF(1,:), state_EKF(2,:), 'LineWidth',1.1);
axis equal; xlim(xlimv); ylim(ylimv);
xlabel('x [m]'); ylabel('y [m]'); title('EKF-SNPID');

% Col 3: Propuesto
nexttile;
plot(xref, yref, 'k--','LineWidth',1.2); hold on; grid on;
plot(state_PRO(1,:), state_PRO(2,:), 'LineWidth',1.1);
axis equal; xlim(xlimv); ylim(ylimv);
xlabel('x [m]'); ylabel('y [m]'); title('Propuesto');

%% ----------------------- FIGURA 2: ERRORES (3 pilares) -----------------------
fig2 = figure('Name','Errores - 3 pilares','NumberTitle','off');
tl2 = tiledlayout(fig2,1,3,'TileSpacing','compact','Padding','compact');
title(tl2,'Errores de seguimiento (e_x, e_y, e_\theta)');

% Límites comunes para errores
err_all = [ex_PID,ey_PID,eth_PID, ex_EKF,ey_EKF,eth_EKF, ex_PRO,ey_PRO,eth_PRO];
eypad = 0.05*max(max(abs(err_all)), eps);
yl_err = [min(err_all)-eypad, max(err_all)+eypad];

% Col 1: PID
nexttile;
plot(t, ex_PID,'LineWidth',1); hold on; grid on;
plot(t, ey_PID,'LineWidth',1);
plot(t, eth_PID,'LineWidth',1);
ylim(yl_err);
xlabel('t [s]'); ylabel('error'); title('PID');
legend('e_x','e_y','e_\theta','Location','best');

% Col 2: EKF-SNPID
nexttile;
plot(t, ex_EKF,'LineWidth',1); hold on; grid on;
plot(t, ey_EKF,'LineWidth',1);
plot(t, eth_EKF,'LineWidth',1);
ylim(yl_err);
xlabel('t [s]'); ylabel('error'); title('EKF-SNPID');

% Col 3: Propuesto
nexttile;
plot(t, ex_PRO,'LineWidth',1); hold on; grid on;
plot(t, ey_PRO,'LineWidth',1);
plot(t, eth_PRO,'LineWidth',1);
ylim(yl_err);
xlabel('t [s]'); ylabel('error'); title('Propuesto');

%% ----------------------- FIGURA 3: CONTROL (3 pilares) -----------------------
fig3 = figure('Name','Control - 3 pilares','NumberTitle','off');
tl3 = tiledlayout(fig3,1,3,'TileSpacing','compact','Padding','compact');
title(tl3,'Acción de control (u_x, u_y, u_\theta)');

% Límites comunes para control (incluir saturación)
u_all = [ux_PID,uy_PID,uth_PID, ux_EKF,uy_EKF,uth_EKF, ux_PRO,uy_PRO,uth_PRO];
uypad = 0.05*max(max(abs(u_all)), eps);
yl_u = [min(u_all)-uypad, max(u_all)+uypad];
yl_u(1) = min(yl_u(1), -sat_limit);
yl_u(2) = max(yl_u(2),  sat_limit);

% Col 1: PID
nexttile;
plot(t, ux_PID,'LineWidth',1); hold on; grid on;
plot(t, uy_PID,'LineWidth',1);
plot(t, uth_PID,'LineWidth',1);
yline(+sat_limit,'--k'); yline(-sat_limit,'--k');
ylim(yl_u);
xlabel('t [s]'); ylabel('u'); title('PID');
legend('u_x','u_y','u_\theta','Location','best');

% Col 2: EKF-SNPID
nexttile;
plot(t, ux_EKF,'LineWidth',1); hold on; grid on;
plot(t, uy_EKF,'LineWidth',1);
plot(t, uth_EKF,'LineWidth',1);
yline(+sat_limit,'--k'); yline(-sat_limit,'--k');
ylim(yl_u);
xlabel('t [s]'); ylabel('u'); title('EKF-SNPID');

% Col 3: Propuesto
nexttile;
plot(t, ux_PRO,'LineWidth',1); hold on; grid on;
plot(t, uy_PRO,'LineWidth',1);
plot(t, uth_PRO,'LineWidth',1);
yline(+sat_limit,'--k'); yline(-sat_limit,'--k');
ylim(yl_u);
xlabel('t [s]'); ylabel('u'); title('Propuesto');


%% ----------------------- METRICAS -----------------------
% Métricas por controlador
metrics_PID  = compute_metrics(t, ex_PID,  ey_PID,  eth_PID,  ux_PID,  uy_PID,  uth_PID,  dt);
metrics_EKF  = compute_metrics(t, ex_EKF,  ey_EKF,  eth_EKF,  ux_EKF,  uy_EKF,  uth_EKF,  dt);
metrics_PRO  = compute_metrics(t, ex_PRO,  ey_PRO,  eth_PRO,  ux_PRO,  uy_PRO,  uth_PRO,  dt);

% Crear tabla
RowNames = {'PID','EKF_SNPID','Propuesto'};
metricsTbl = table( ...
    [metrics_PID.RMSE_pos; metrics_EKF.RMSE_pos; metrics_PRO.RMSE_pos], ...
    [metrics_PID.RMSE_x;   metrics_EKF.RMSE_x;   metrics_PRO.RMSE_x], ...
    [metrics_PID.RMSE_y;   metrics_EKF.RMSE_y;   metrics_PRO.RMSE_y], ...
    [metrics_PID.RMSE_th;  metrics_EKF.RMSE_th;  metrics_PRO.RMSE_th], ...
    [metrics_PID.MAE_pos;  metrics_EKF.MAE_pos;  metrics_PRO.MAE_pos], ...
    [metrics_PID.Max_epos; metrics_EKF.Max_epos; metrics_PRO.Max_epos], ...
    [metrics_PID.IAE;      metrics_EKF.IAE;      metrics_PRO.IAE], ...
    [metrics_PID.ISE;      metrics_EKF.ISE;      metrics_PRO.ISE], ...
    [metrics_PID.U_L1;     metrics_EKF.U_L1;     metrics_PRO.U_L1], ...
    [metrics_PID.U_L2;     metrics_EKF.U_L2;     metrics_PRO.U_L2], ...
    [metrics_PID.max_ux;   metrics_EKF.max_ux;   metrics_PRO.max_ux], ...
    [metrics_PID.max_uy;   metrics_EKF.max_uy;   metrics_PRO.max_uy], ...
    [metrics_PID.max_uth;  metrics_EKF.max_uth;  metrics_PRO.max_uth], ...
    [metrics_PID.pct_within_tol; metrics_EKF.pct_within_tol; metrics_PRO.pct_within_tol], ...
    'VariableNames', {'RMSE_pos','RMSE_x','RMSE_y','RMSE_th', ...
                      'MAE_pos','Max_e_pos','IAE','ISE', ...
                      'U_L1','U_L2','max|ux|','max|uy|','max|uth|','PctWithinTol'}, ...
    'RowNames', RowNames);

disp('==== METRICAS COMPARATIVAS (Trayectoria Rosa) ====');
disp(metricsTbl);

%% ================== FUNCIONES AUXILIARES ==================
function [u_sat, sum_e_new] = sat_aw(u, sum_e, e, umax)
    % Saturación + corrección simple de integral (back calculation light)
    if abs(u) > umax
        u_sat = sign(u)*umax;
        sum_e_new = sum_e - e;
    else
        u_sat = u;
        sum_e_new = sum_e;
    end
end

function [u_sat, w_new, P_new] = ekf_snpid(x1,x2,x3, w, P, Q, R, n, alpha, umax)
    x = [x1; x2; x3];
    v = w' * x;
    u = alpha * tanh(v);
    u_sat = max(-umax, min(umax, u));
    es    = u_sat - u;       % error por saturación
    if abs(es)>0
        e_aw = x1 + es;      % innovación simple con back-calc
    else
        e_aw = x1;
    end
    sech2 = (1 - tanh(v)^2);
    H = alpha * sech2 * x;   % Jacobiano wrt w
    S = H' * P * H + R;
    K = (P * H) / S;
    P_new = P - K * H' * P + Q;
    w_new = w + (n .* K) * e_aw;
end

function [u_sat, w_new, P_new] = ekf_snpid_freeze(x1,x2,x3, w, P, Q, R, n, alpha, umax, freeze_flag)
    if freeze_flag
        % Salida congelada con pesos actuales (sin update)
        x = [x1; x2; x3];
        v = w' * x;
        u = alpha * tanh(v);
        u_sat = max(-umax, min(umax, u));
        w_new = w; P_new = P;
        return;
    end
    % Igual que ekf_snpid si no está congelado
    [u_sat, w_new, P_new] = ekf_snpid(x1,x2,x3, w, P, Q, R, n, alpha, umax);
end

function ang = wrapToPi_local(ang)
    ang = mod(ang + pi, 2*pi) - pi;
end

function M = compute_metrics(t, ex, ey, eth, ux, uy, uth, dt)
    epos = sqrt(ex.^2 + ey.^2);
    M.RMSE_x  = sqrt(mean(ex.^2));
    M.RMSE_y  = sqrt(mean(ey.^2));
    M.RMSE_th = sqrt(mean(eth.^2));
    M.RMSE_pos= sqrt(mean(epos.^2));
    M.MAE_pos = mean(epos);
    M.Max_epos= max(epos);

    M.IAE = sum(abs(ex)+abs(ey)+abs(eth)) * dt;
    M.ISE = sum(ex.^2 + ey.^2 + eth.^2)  * dt;

    M.U_L1 = sum(abs(ux)+abs(uy)+abs(uth)) * dt;
    M.U_L2 = sum(ux.^2 + uy.^2 + uth.^2)  * dt;

    M.max_ux = max(abs(ux));
    M.max_uy = max(abs(uy));
    M.max_uth= max(abs(uth));

    % % dentro de tolerancias (útil en seguimiento): 5 cm y 5°
    tol_pos = 0.05;    % m
    tol_th  = 5*pi/180;% rad
    within = (epos < tol_pos) & (abs(eth) < tol_th);
    M.pct_within_tol = 100 * sum(within)/numel(t);
end
