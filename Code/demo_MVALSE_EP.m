% This code is written by Ning Zhang and Jiang Zhu. If you have any
% problems, please feel free to contact jiangzhu16@zju.edu.cn
% demo for the MVALSE-EP
clear;
close all;
clc;  
rng(1)
N    = 100;  
M    = 100;     
T = 5;
sigma = 1;
SNR_min = 0;
Delta_dB = 5;  % 15
Mcal = (0:N-1)';    
Iter_max = 100;

% DOAs_grid = [-3;2;75];
DOAs_grid = [-30;-3;2;30;75];
Ktrue    = length(DOAs_grid);    
B_all = [1;3;5];

NMSE = zeros(Iter_max,length(B_all));
tic
bias = (2*rand(Ktrue,1)-1)/4;
DOAs = DOAs_grid+bias;
coeff_trans_DOAs = 1./(pi*cosd(DOAs))*180/pi;
omega = pi*sind(DOAs);
A   = exp(1j*(0:1:Mcal(end)).'*omega.');
phase = exp(1j*2*pi*rand(Ktrue,T));
Noise = sqrt(sigma/2)*(randn(N,T)+1j*randn(N,T));
Fluc_dB = rand(Ktrue,1);
SNR_dB = SNR_min+Delta_dB*Fluc_dB;
gain = sqrt(10.^(SNR_dB./10)*sigma);
X = bsxfun(@times,gain,phase);
sig_var = Ktrue*mean(abs(gain).^2)+sigma;
Z = A*X;

gain_true_ave = mean(abs(X).^2,2);


Y = Z + Noise;
Y_M = Y(Mcal+1,:);
for bit_idx = 1:length(B_all)
    B = B_all(bit_idx);
    num_bins = 2^B;
    % use the dynamic range to design the quantizer
    Delta_max = 3*sqrt(sig_var/2);
    Delta_min = -Delta_max;
    step_intval = (Delta_max - Delta_min)/(num_bins);
    Y_R = floor((real(Y_M)-Delta_min)/step_intval);
    Y_R(real(Y_M)>=Delta_max) = num_bins-1;
    Y_R(real(Y_M)<=Delta_min) = 0;
    Y_I = floor((imag(Y_M)-Delta_min)/step_intval);
    Y_I(imag(Y_M)>=Delta_max) = num_bins-1;
    Y_I(imag(Y_M)<=Delta_min) = 0;
    Y_R_q = Delta_min + (Y_R+0.5)* step_intval;
    Y_I_q = Delta_min + (Y_I+0.5)* step_intval;
    y_vec_RI = [Y_R(:);Y_I(:)];
    % quantized observations
    Y_q = Y_R_q+1j*Y_I_q;
    Y_index = Y_R+1j*Y_I;
    if B == 1
        Est_noisevar_not = 0;
    else
        Est_noisevar_not = 1;
    end

    result_all = MVALSE_EP(Y_q, Mcal, Z, y_vec_RI,...
        Delta_min, B, step_intval, Iter_max, sigma, Est_noisevar_not);
    weight_est = mean(abs(result_all.weight).^2,2);
    if B == 1
        weight_est = weight_est/max(weight_est)*max(gain_true_ave);
    end
    figure(bit_idx)
    polarplot(omega,gain_true_ave,'ro')
    hold on
    polarplot(result_all.freqs,weight_est,'b+')
    legend('Truth',['Est (B=',num2str(B),')'],'Fontsize',14)
    NMSE(:,bit_idx) = result_all.NMSE_Z;

end
toc

lw = 1.6;
fsz = 14;
msz = 6;
iter_num = [1;3;5;7;10;20;30;40;50];
% iter_num = 1:Iter_max-1;
figure()
plot(iter_num,10*log10(NMSE(iter_num,1)),'-rs','LineWidth',lw,'MarkerSize',msz)
hold on
plot(iter_num,10*log10(NMSE(iter_num,2)),'-bo','LineWidth',lw,'MarkerSize',msz)
plot(iter_num,10*log10(NMSE(iter_num,3)),'-md','LineWidth',lw,'MarkerSize',msz)
legend(['B=',num2str(B_all(1))],...
    ['B=',num2str(B_all(2))],...
    ['B=',num2str(B_all(3))],...
    'Fontsize',fsz)
xlabel('${\rm SNR}_{\rm min}$ dB','Interpreter','latex','FontSize',fsz)
ylabel('${\rm NMSE}(\hat{\mathbf Z})$ (dB)','Interpreter','latex','FontSize',fsz)
