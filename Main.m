clear all,
close all
clc
%% data
% noise_var = 1;
% Noise = sqrt(noise_var)*randn(1,1e5);
% hist(Noise)
% hold on
% % Noise = sqrt(noise_var)*(2*rand(1,10)-1);
% u = sqrt(noise_var)*( 2*(randn(1,1e5)>0)-1 );
% hist(u)

mu_vec = linspace( 1e-3, 10e-3,10 );
length_mu_vec = length(mu_vec);
[ mse,thy ] = deal(zeros(length_mu_vec,1));
% Monter Carlo
MC = 500;
% Data section
[ inputDimension, trainSize, testSize ] = deal( 10, 0.7e4, 0 );
[ input_var, noise_var ] = deal( 1, 1e-1 );
% nonstationary
plant_var = 1e-6;
% % stationary
% plant_var = 0;
generatenumber = inputDimension*trainSize;
h = waitbar( 0,'Curves are generating!' );
% set of LMMN
[ delte ] = deal( 0.5432 );
Xi_4 = 3*(noise_var^2);
Xi_6 = 15*(noise_var^3);
delte_complement = 1 - delte;
a = delte*delte*noise_var + 2*delte*delte_complement*Xi_4 + delte_complement*delte_complement*Xi_6;
b = delte + 3*delte_complement*noise_var;
c = delte*delte + 12*delte*delte_complement*noise_var + 15*delte_complement*Xi_4;
%% algorithm
% Choose a different step size
for i = 1:length_mu_vec
    stepSizeWeightVector = mu_vec(i);
    Sum_learningCurve_LMMN = 0;
    % MC Simulation
    for mc = 1:MC
        display(mc)
        % Data source
        Filter_w = (1/sqrt(inputDimension))*ones(1,inputDimension);
        out_first = 1;
        Inputsignal = sqrt(input_var)*randn(1,generatenumber);
        % get trainInput
        [trainInput,~,~,~] = distribution(Inputsignal,zeros(1,generatenumber),inputDimension,trainSize,0);
%         desired_sig_cle = filter(Filter_w,out_first,Inputsignal);
%         % Noise source
%         u = sqrt(noise_var)*( 2*( randn(1,length(Inputsignal))>0 )-1 );
        Noise = sqrt(noise_var)*randn(1,length(Inputsignal));%because the size of inputsignal is equal the size of disired_sig_cle
%         desired_sig_noise = desired_sig_cle+Noise;
%         % Function section
% %         tic;
%         [~,trainTarget,~,~] = distribution(zeros(1,generatenumber),desired_sig_noise,inputDimension,trainSize,0);
        [ learningCurve, MSE1, MSE2 ]= deal(zeros(trainSize,1));
        weightVector = zeros(inputDimension,1);
        % training
        for n = 1:trainSize
            % nonstationary environment
            q = sqrt(plant_var)*randn(1)*ones(1,inputDimension);
            Filter_w = Filter_w + q;
            reversal = trainInput(:,n)';
            x1 = filter(Filter_w,out_first,reversal);
            y = x1(:,inputDimension);            
            desired_sig_cle = y;
            desired_sig_noise = desired_sig_cle+Noise(n);
            
            error = desired_sig_noise -  weightVector'*trainInput(:,n);
            errFunction = delte*error + delte_complement*(error^3);
            weightVector = weightVector + stepSizeWeightVector*errFunction*trainInput(:,n);
            aprioriErr = (Filter_w - weightVector)'*trainInput(:,n);
            learningCurve(n) = mean(aprioriErr.^2);
       end
        Sum_learningCurve_LMMN = Sum_learningCurve_LMMN+learningCurve;
%        toc;
    end
    %% plot
    Aver_learningCurve_LMMN = Sum_learningCurve_LMMN/MC;
   %figure section
    figure(1)
    plot(10*log10(Aver_learningCurve_LMMN),'-.r','LineWidth',2);hold on;
    mse(i) = mean(Aver_learningCurve_LMMN(end-200:end));
    trR_part = mu_vec(i)*inputDimension*input_var*a;
    trQ_part = (mu_vec(i)^(-1))*inputDimension*plant_var;
    denominator_part = 2*b - mu_vec(i)*( inputDimension + 2 )*input_var*c;
    thy(i) = ( trR_part + trQ_part )/denominator_part;
    waitbar( i/length_mu_vec)                    
end
close(h)
%% plot
%figure section
figure(2)
plot(mu_vec,mse,'r:<','LineWidth',2);hold on;
plot(mu_vec,thy,'b-o','LineWidth',2);hold on;
% axis
xlabel('iteration','FontName','Times New Roman','FontSize',20); 
ylabel('EMSE','FontName','Times New Roman','FontSize',20); 
set(gca,'FontSize',18);
legend('Simulation','Theory')
title(' Experimental and theoretical EMSE versus \mu for LMMN');