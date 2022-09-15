function result_all = MVALSE_EP(Y_q, Mcal, Z, y_vec_RI,...
    Delta_min, B, step_intval, Iter_max, sigma_true, Est_noisevar_not)

    [M,T] = size(Y_q);
    y_vec_R = y_vec_RI(1:end/2);
    y_vec_I = y_vec_RI(end/2+1:end);
    Y_R = reshape(y_vec_R,M,T);
    Y_I = reshape(y_vec_I,M,T);
    Y_RI = Y_R+1j*Y_I;

    
    N = Mcal(end)+1;

    % MMSE calculation
    Z_A_ext = zeros(M,T);
    V_A_ext = var(Y_q(:))*ones(M,T);
    z_A_ext_vec = Z_A_ext(:);
    v_A_ext_vec = V_A_ext(:);
    z_A_ext_vec_RI = [real(z_A_ext_vec);imag(z_A_ext_vec)];
    v_A_ext_vec_RI = [v_A_ext_vec;v_A_ext_vec]/2;

    % initialize sigma_hat
    if B == 1
        sigma_hat = sigma_true;
    else
        sigma_hat = var(Y_q(:));
    end

    var_min = 1e-14;
    var_max = 1e14;

    sigma_n0_ext_RI = sigma_hat*ones(2*M*T,1)/2;
    [z_B_post_vec_RI, v_B_post_vec_RI] = GaussianMomentsComputation_MJH...
            (y_vec_RI, z_A_ext_vec_RI, v_A_ext_vec_RI, Delta_min, B, step_intval,...
            sigma_n0_ext_RI);
    
    z_B_post_vec = z_B_post_vec_RI(1:end/2)+1j*z_B_post_vec_RI(end/2+1:end);
    v_B_post_vec = v_B_post_vec_RI(1:end/2)+v_B_post_vec_RI(end/2+1:end);
    Z_B_post = reshape(z_B_post_vec,M,T);
    V_B_post = reshape(v_B_post_vec,M,T);
    
    V_B_ext = V_B_post.*V_A_ext./(V_A_ext-V_B_post);
    V_B_ext = max(V_B_ext,var_min);
    V_B_ext = min(V_B_ext,var_max);
    Z_B_ext = V_B_ext.*(Z_B_post./V_B_post-Z_A_ext./V_A_ext);

    
    % initialization
    Y_tilde = Z_B_ext;
    Sigma_tilde = V_B_ext;
    Sigma_tilde_all = ones(N,T);
    Sigma_tilde_all(Mcal+1,:) = Sigma_tilde;
    Res   = Y_tilde;
    
    L = N;
    mu_all = zeros(L,1);
    kappa_all = ones(L,1);
    A     = zeros(L,L);
    J     = zeros(L,L,T);
    h     = zeros(L,T);
    w     = zeros(L,T);
    C     = zeros(L,L,T);

    iter = 1;
    Khat = round(0.4995*L);
    tau = max((Y_tilde(:)'*Y_tilde(:)-sum(sum(Sigma_tilde)))/Khat/M,var(Y_q(:))/Khat);
    rho = Khat/L;
    NMSE_Z = zeros(Iter_max,1);
    Kt = zeros(Iter_max,1);


    

    Zro = zeros(N,T);
    for l = 1:L
        yI = zeros(N,T);
        yI(Mcal+1,:) = Res;
        den_noise = ones(N,1)*sum(1./Sigma_tilde);
        R = (yI./Sigma_tilde_all./den_noise)*(yI./Sigma_tilde_all)';
        sR = zeros(N-1,1);
        
        % noncoherent estimation of the pdf
        for i = 2:N
            for k=1:i-1
                sR(i-k) = sR(i-k) + R(i,k);
            end
        end
        etaI   = 2*sR; 
        ind    = find(abs(etaI)>0);
    
        [~,mu,kappa] = Heuristic2(etaI(ind), ind);
        mu_all(l) = mu;
        kappa_all(l) = kappa;
        A(Mcal+1,l) = exp(1i*Mcal * mu) .* ( besseli(Mcal,kappa,1)/besseli(0,kappa,1) );
        
        % compute weight estimates; rank one update
        w_temp = w(1:l-1,:);   % (l-1)*T
        C_temp = C(1:l-1,1:l-1,:);
        J(1:l-1,l,:) = reshape(A(Mcal+1,1:l-1)'*(A(Mcal+1,l)./Sigma_tilde),l-1,1,T); 
        J(l,1:l-1,:)  = pagectranspose(J(1:l-1,l,:));
        J(l,l,:) = reshape(sum(1./Sigma_tilde),1,1,T);
        h(l,:) = A(Mcal+1,l)'*(Y_tilde./Sigma_tilde);
        CJ = pagemtimes(C_temp,J(1:l-1,l,:));
        JhCJ = real(reshape(pagemtimes(J(1:l-1,l,:),'ctranspose',CJ,'none'),1,T));
        v = 1./(sum(1./Sigma_tilde)+1/tau-JhCJ);   % 1*T
        w_temp_tensor = reshape(w_temp,l-1,1,T);
        temp_Jw = reshape(pagemtimes(J(1:l-1,l,:),'ctranspose',w_temp_tensor,'none'),1,T);
        u = v .* (h(l,:) - temp_Jw);  % 1*T
        w(l,:) = u;
        ctemp = reshape(pagemtimes(C_temp,J(1:l-1,l,:)),l-1,T);   % (l-1)*T
        w(1:l-1,:) = w_temp - bsxfun(@times,ctemp,u);
        ctemp_tensor = reshape(ctemp,l-1,1,T);
        ctemp_tensor_outprod = pagemtimes(ctemp_tensor,'none',ctemp_tensor,'ctranspose');
        v_tensor = reshape(v,1,1,T);
        C(1:l-1,1:l-1,:) = C_temp + bsxfun(@times,v_tensor, ctemp_tensor_outprod);
        C(1:l-1,l,:) = reshape(-bsxfun(@times,v,ctemp),l-1,1,T);  
        C(l,1:l-1,:) = pagectranspose(C(1:l-1,l,:)); 
        C(l,l,:) = reshape(v,1,1,T);
        % the residual signal
        Res = Y_tilde - A(Mcal+1,1:l)*w(1:l,:);
        
        if l==Khat % save mse and K at initialization
            Zr    = A(:,1:l)*w(1:l,:);
            if B>1
                deb_z = 1;
            else
                % diag case
                deb_z = diag(Zr'*Z)./(diag(Zr'*Zr)+eps);
            end
            NMSE_Z(iter) = norm(Zr*diag(deb_z)-Z,'fro')^2/norm(Z,'fro')^2;
            Kt(iter)  = Khat;
        end
    end
    
    C_ave = mean(C,3);
    tau = real( sum(sum(conj(w(1:Khat,:)).*w(1:Khat,:)))/T+trace(C_ave(1:Khat,1:Khat)) )/Khat;% sigma_tilde = nu*ones(M,1);

    while true
        iter = iter+1;
        % Step 1: update the support s and weight
        [ Khat, s, w, C ] = maxZ_diag( J, h, M, Sigma_tilde, rho, tau );
        % update the model parameters
        if Khat>0
            C_ave = mean(C,3);
            tau = real( sum(sum(conj(w(s,:)).*w(s,:)))/T+trace(C_ave(s,s)) )/Khat;
            if Khat<L
                rho = Khat/L;
            else
                % just to avoid the potential issue of log(1-rho) when rho=1
                rho = max((L-1)/L,0.99); 
            end

        else
            % just to avoid the potential issue of log(rho) when rho=0
            rho = 1/L; 
        end
        % Step 2: Update the posterior pdfs of the frequencies
        inz = 1:L; inz = inz(s); % indices of the non-zero components
        th = zeros(Khat,1);
    %     kappa_est = zeros(K,1);
        thetaz = mu_all(inz);
        for i = 1:Khat
            if Khat == 1
                eta = sum(2*(Y_tilde./Sigma_tilde)*w(inz,:)',2);
            else
                A_i = A(Mcal+1,inz([1:i-1 i+1:end]));
                r = Y_tilde - A_i*w(inz([1:i-1 i+1:end]),:);
                eta_all = 2*(1./Sigma_tilde).* ( r * w(inz(i),:)' - A_i * reshape(C(inz([1:i-1 i+1:end]),i,:),Khat-1,T) );
                eta = sum(eta_all,2);
            end
            
            [A(:,inz(i)), th(i), kappa_est] = Heuristic2_warm_start( eta, Mcal , thetaz(i));
            mu_all(inz(i)) = th(i);
            kappa_all(inz(i)) = kappa_est;
        end
    
        % Step 2: obtain the posterior means and variances of Z (Mcal) in module A
        Z_A_post     = A(Mcal+1,s)*w(s,:);
    
        A_Mcal_tensor = repmat(A(Mcal+1,s),[1 1 T]);
        CAh = pagemtimes(C(s,s,:),'none',A_Mcal_tensor,'ctranspose');
        post_var_fast_1 = real(squeeze(sum(A_Mcal_tensor.*pagetranspose(CAh),2)));
        post_var_fast_2 = ones(M,1)*sum(w(s,:).*conj(w(s,:)),1)-abs(A(Mcal+1,s)).^2*(abs(w(s,:)).^2);
        C_sub3 = reshape(C(s,s,:),[],T);
        Selector = diag(true(size(C(s,s,:),1),1));
        C_sub_mat = C_sub3(Selector(:),:);
        post_var_fast_3 = ones(M,1)*sum(C_sub_mat,1)-abs(A(Mcal+1,s)).^2*C_sub_mat;
        V_A_post = post_var_fast_1+post_var_fast_2+post_var_fast_3;
    
    
        V_A_post = real(V_A_post);
    
        
        Zr = A(:,s)*w(s,:);
        if B>1
            deb_z = 1;
        else
            % diag case
            deb_z = diag(Zr'*Z)./(diag(Zr'*Zr)+eps);
        end
        NMSE_Z(iter) = norm(Zr*diag(deb_z)-Z,'fro')^2/norm(Z,'fro')^2;
        Kt(iter)  = Khat;
        V_A_ext = real(V_A_post.*Sigma_tilde./(Sigma_tilde-V_A_post));
        V_A_ext = max(V_A_ext,var_min);
        V_A_ext = min(V_A_ext,var_max);

        Z_A_ext = (Z_A_post./V_A_post-Y_tilde./Sigma_tilde).*V_A_ext;
    
        % Step 2: perform the MMSE of Z in module B
    %     V_A_ext = V_A_ext.*(V_A_ext>=0)+(V_A_ext<0)*var_max;
        z_A_ext_vec = Z_A_ext(:);
        v_A_ext_vec = V_A_ext(:);
        z_A_ext_vec_RI = [real(z_A_ext_vec);imag(z_A_ext_vec)];
        v_A_ext_vec_RI = [v_A_ext_vec;v_A_ext_vec]/2;

        sigma_n0_ext_RI = sigma_hat*ones(2*M*T,1)/2;
        [z_B_post_vec_RI, v_B_post_vec_RI] = GaussianMomentsComputation_MJH...
                (y_vec_RI, z_A_ext_vec_RI, v_A_ext_vec_RI, Delta_min, B, step_intval,...
                sigma_n0_ext_RI);
        
        z_B_post_vec = z_B_post_vec_RI(1:end/2)+1j*z_B_post_vec_RI(end/2+1:end);
        v_B_post_vec = v_B_post_vec_RI(1:end/2)+v_B_post_vec_RI(end/2+1:end);
        Z_B_post = reshape(z_B_post_vec,M,T);
        V_B_post = reshape(v_B_post_vec,M,T);

        % average
%         V_B_post = mean(V_B_post,2)*ones(1,T);
        V_B_ext = V_B_post.*V_A_ext./(V_A_ext-V_B_post);
        V_B_ext = max(V_B_ext,var_min);
        V_B_ext = min(V_B_ext,var_max);

        Z_B_ext = V_B_ext.*(Z_B_post./V_B_post-Z_A_ext./V_A_ext);
    
    
        % Step 3: obtain the pseudo observations of Z in standard linear model
    %     V_B_ext = V_B_ext.*(V_B_ext>=0)+(V_B_ext<0)*var_max;
        Sigma_tilde = V_B_ext;
        Y_tilde = Z_B_ext;
%         sigma_hat = norm(Y_tilde-Z_B_post,'fro')^2/(M*T)+mean(V_B_post(:));
        if Est_noisevar_not
            sigma_hat = norm(Y_tilde-Z_A_post,'fro')^2/(M*T)+mean(V_A_post(:));
        end

        
        % Step 4: update J and h and go to Step 1 (all must be updated)
        A_Mcal_tensor = repmat(A(Mcal+1,:),[1 1 T]);
        Sigma_tilde_tensor = reshape(Sigma_tilde,M,1,T);
        Sigma_inv_A = bsxfun(@rdivide,A_Mcal_tensor,Sigma_tilde_tensor);
        J_fast = pagemtimes(A_Mcal_tensor,'ctranspose',Sigma_inv_A,'none');
        J_fast3 = reshape(J_fast,[],T);
        Selector = diag(true(size(J_fast(:,:,:),1),1));

        J_fast3(Selector(:),:) =  ones(L,1)*sum(1./Sigma_tilde);
        J = reshape(J_fast3,L,L,T);
        h  = A(Mcal+1,:)'*(Y_tilde./Sigma_tilde);
 
        if (norm(Zro-Zr)/norm(Zr)<1e-6) || (norm(Zr)==0&&norm(Zro-Zr)==0) || (iter >= Iter_max)
            NMSE_Z(iter+1:end) = NMSE_Z(iter);
            Kt(iter+1:end)  = Kt(iter);
            break;
        end
        Zro = Zr;
    end
    
    result_all = struct('freqs',mu_all(inz),'concentrationparam',...
        kappa_all(inz),'Kt',Kt,'NMSE_Z',NMSE_Z,'weight',w(s,:));
end
    
function [ K, s, w, C ] = maxZ_diag( J, h, M, sigma, rho, tau )
    %maxZ maximizes the function Z of the binary vector s, see Appendix A of
    % the paper
    
    [L,G] = size(h);
    cnst = log(rho/(1-rho))+log(1/tau);
    
    K = 0; % number of components
    s = false(L,1); % Initialize s
    w = zeros(L,G);
    C = zeros(L,L,G);
    u = zeros(L,G);
    v = zeros(L,G);
    Delta_MMV = zeros(L,G);
    if L > 1
        cont = 1;
        while cont
            if K<M-1
                C_temp  = pagemtimes(C(s,s,:),J(s,~s,:));
                JhCJ = real(squeeze(sum(J(s,~s,:).*conj(C_temp),1)));
                v(~s,:) = 1./(sum(1./sigma)+1/tau-JhCJ);   % 1*T
%                 v(~s,g) = 1 ./ ( sum(1./sigma) + 1/tau - real(sum(J(s,~s,g).*conj(C(s,s,g)*J(s,~s,g)),1)) ); % diagonal
                w_temp = reshape(w(s,:),sum(s),1,G);
                Jtw = squeeze(pagemtimes(J(s,~s,:),'ctranspose',w_temp,'none'));
                u(~s,:) = v(~s,:) .* ( h(~s,:) - Jtw);
                Delta_MMV(~s,:) = log(v(~s,:)) + u(~s,:).*conj(u(~s,:))./v(~s,:) + cnst;
            else
                Delta_MMV(~s,:) = -1; % dummy negative assignment to avoid any activation
            end
            if ~isempty(h(s,:))
                C_sub3 = reshape(C(s,s,:),[],G);
                Selector = diag(true(size(C(s,s,:),1),1));
                C_sub_mat = C_sub3(Selector(:),:);
                Delta_MMV(s,:) = -log(C_sub_mat) - w(s,:).*conj(w(s,:))./C_sub_mat - cnst;
            end
            Delta = sum(Delta_MMV,2);
            [~, k] = max(Delta);
            if Delta(k)>0
                if s(k)==0 % activate
                    w(k,:) = u(k,:);
                    ctemp_tensor = pagemtimes(C(s,s,:),J(s,k,:));
                    ctemp = reshape(ctemp_tensor,sum(s),G);
                    w(s,:) = w(s,:) - ctemp.*u(k,:);
                    ctemp_tensor_outprod = pagemtimes(ctemp_tensor,'none',ctemp_tensor,'ctranspose');
                    v_tensor = reshape(v(k,:),1,1,G);
                    C(s,s,:) = C(s,s,:) + bsxfun(@times,v_tensor, ctemp_tensor_outprod);

                    C(s,k,:) = -pagemtimes(v_tensor,ctemp_tensor);
                    C(k,s,:) = pagectranspose(C(s,k,:));
                    C(k,k,:) = v(k,:);

                    s(k) = ~s(k); K = K+1;
                else % deactivate
                    s(k) = ~s(k); K = K-1;
                    w(s,:) = w(s,:) - reshape(C(s,k,:),sum(s),G).*w(k,:)./reshape(C(k,k,:),1,G);

                    C(s,s,:) = C(s,s,:) - C(s,k,:).*C(k,s,:)./C(k,k,:);
                end
                C = (C+pagectranspose(C))/2;
            else
                break
            end
        end
        % L is not equal to 1
    elseif L == 1
        if s == 0
            v = 1 ./ ( sum(1./sigma) + 1/tau );  % 1*T
            u = v.* h;
            Delta = mean(log(v) + u.*conj(u)/v + cnst);


            
            
            if Delta>0
                w = u;
                C(1,1,:) = v; s = 1; K = 1;

            end
        else
            C_mat = reshape(C,1,G);
            Delta = mean(-log(C_mat) - w.*conj(w)./C_mat - cnst);

            if Delta>0
                w = 0; C = 0; s = 0; K = 0;
            end
        end
    end
end
function [a, theta, kappa] = Heuristic2_warm_start( eta, m , mu)
    %Heuristic2 Uses the mixture of von Mises approximation of frequency pdfs
    %and Heuristic #2 to output one von Mises pdf
    % m = (0:1:N-1)';
    N = m(end)+1;
    m_sym = m-(N-1)/2;
    eta_sym = eta*exp(-1i*(N-1)/2*mu);
    d1_sym     = -imag( eta_sym' * ( m_sym    .* exp(1i*m_sym*mu) ) );
    d2_sym     = -real( eta_sym' * ( m_sym.^2 .* exp(1i*m_sym*mu) ) );
    if d2_sym<0 % if the function is locally concave (usually the case)
        theta  = mu - d1_sym/d2_sym;
        kappa  = Ainv( exp(0.5/d2_sym) );
    else    % if the function is not locally concave (not sure if ever the case)
        theta  = mu;
    %     kappa  = abs(eta_q(in));
        kappa  =1e2;
    end

    n      = (0:1:m(end))';
    a      = exp(1i*n * theta).*( besseli(n,kappa,1)/besseli(0,kappa,1) );
end


function [a, theta, kappa] = Heuristic2( eta, m )
    %Heuristic2 Uses the mixture of von Mises approximation of frequency pdfs
    %and Heuristic #2 to output one von Mises pdf
    
    N     = length(m);
    ka    = abs(eta);
    A     = besseli(1,ka,1)./besseli(0,ka,1);
    kmix  = Ainv( A.^(1./m.^2) );
    k     = N;
    eta_q = kmix(k) * exp( 1i * ( angle(eta(k)) + 2*pi*(1:m(k)).' )/m(k) );
    for k = N-1:-1:1
        if m(k) ~= 0
            phi   = angle(eta(k));
            eta_q = eta_q + kmix(k) * exp( 1i*( phi + 2*pi*round( (m(k)*angle(eta_q) - phi)/2/pi ) )/m(k) );
        end
    end
    [~,in] = max(abs(eta_q));
    mu     = angle(eta_q(in));
    d1     = -imag( eta' * ( m    .* exp(1i*m*mu) ) );
    d2     = -real( eta' * ( m.^2 .* exp(1i*m*mu) ) );
    if d2<0 % if the function is locally concave (usually the case)
%         theta  = mu - d1/d2;
%         kappa  = Ainv( exp(0.5/d2) );
        theta_1  = mu - d1/d2;
    %     kappa_1  = Ainv( exp(0.5/d2) );
        
        theta_ave = (theta_1+mu)/2;
        d1_ave     = -imag( eta' * ( m    .* exp(1i*m*theta_ave) ) );
        d2_ave     = -real( eta' * ( m.^2 .* exp(1i*m*theta_ave) ) );
        theta  = theta_ave - d1_ave/d2_ave;
        kappa  = Ainv( exp(0.5/d2_ave) );

    else    % if the function is not locally concave (not sure if ever the case)
        theta  = mu;
        kappa  = abs(eta_q(in));
    end
    n      = (0:1:m(end))';
    a      = exp(1i*n * theta).*( besseli(n,kappa,1)/besseli(0,kappa,1) );
end
function [ k ] = Ainv( R )
    % Returns the approximate solution of the equation R = A(k),
    % where A(k) = I_1(k)/I_0(k) is the ration of modified Bessel functions of
    % the first kind of first and zero order
    % Uses the approximation from
    %       Mardia & Jupp - Directional Statistics, Wiley 2000, pp. 85-86.
    %
    % When input R is a vector, the output is a vector containing the
    % corresponding entries

    k   = R; % define A with same dimensions
    in1 = (R<.53); % indices of the entries < .53
    in3 = (R>=.85);% indices of the entries >= .85
    in2 = logical(1-in1-in3); % indices of the entries >=.53 and <.85
    R1  = R(in1); % entries < .53
    R2  = R(in2); % entries >=.53 and <.85
    R3  = R(in3); % entries >= .85

    % compute for the entries which are < .53
    if ~isempty(R1)
        t      = R1.*R1;
        k(in1) = R1 .* ( 2 + t + 5/6*t.*t );
    end
    % compute for the entries which are >=.53 and <.85
    if ~isempty(R2)
        k(in2) = -.4 + 1.39*R2 + 0.43./(1-R2);
    end
    % compute for the entries which are >= .85
    if ~isempty(R3)
        k(in3) = 1./( R3.*(R3-1).*(R3-3) );
    end
end
