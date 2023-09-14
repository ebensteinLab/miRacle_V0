
C = [2465, 104, 20 , 60 ; 
     83  , 570, 1  , 2  ;
     43  , 1  , 94 , 5  ;
     190 , 3  , 9  , 414];
 
N_trials = 10000; 
p_vec = C(:) / sum(C(:));
multinomial = makedist('Multinomial',p_vec);

for n = 1:N_trials
rand_samples1 = multinomial.random(sum(C(:)),1);
rand_samples2 = multinomial.random(sum(C(:)),1);
rand_samples3 = multinomial.random(sum(C(:)),1);

C_new_flat1 = zeros(16,1);
C_new_flat2 = zeros(16,1);
C_new_flat3 = zeros(16,1);

for ii = 1:16
    C_new_flat1(ii)=sum(rand_samples1==ii);
    C_new_flat2(ii)=sum(rand_samples2==ii);
    C_new_flat3(ii)=sum(rand_samples3==ii);

end
C_new1 = reshape(C_new_flat1,4,4);
C_new2 = reshape(C_new_flat2,4,4);
C_new3 = reshape(C_new_flat3,4,4);

C_n1 = C_new1;
C_n2 = C_new2;
C_n3 = C_new3;

for ii = 1:4
    C_n1(ii,:) = C_n1(ii,:)/sum(C_n1(ii,:));
    C_n2(ii,:) = C_n2(ii,:)/sum(C_n2(ii,:));
    C_n3(ii,:) = C_n3(ii,:)/sum(C_n3(ii,:));

end
counts_111 = [137680 30254 7008 23021 ];
counts_253 = [150738 25625 11899 25969];
counts_100 = [73374 36678 424 1141];


counts_111 = counts_111.* (1+0.05*[randn(1,4)]);
counts_253 = counts_253.* (1+0.05*[randn(1,4)]);
counts_111_2 = counts_111.* (1+0.05*[randn(1,4)]);

counts_111 = counts_111 * C_n1^-1 ;
counts_253 = counts_253 * C_n2^-1 ;
counts_111_2 = counts_111_2 * C_n3^-1 ;

r1 = counts_111(2:end); r1 = r1/sum(r1);
r2 = counts_253(2:end); r2 = r2/sum(r2);
r3 = counts_111_2(2:end); r3 = r3/sum(r3);


Z = r2./r1 / sum(r2./r1);
Z2 = r3./r1 / sum(r3./r1);

rgb_mat(n,:)=Z;
rgb111_mat(n,:)=Z2;

end
rgb_mat = rgb_mat(logical(prod(rgb_mat>0,2)),:);
rgb_mat_orig = rgb_mat;
% plot3([1,0],[0,1],[0,0],'k-','linewidth',3); hold on;
% plot3([1,0],[0,0],[0,1],'k-','linewidth',3);
% plot3([0,0],[1,0],[0,1],'k-','linewidth',3);
% xlabel('15b');
% ylabel('155');
% zlabel('126');
% scatterplot=plot3(rgb_mat(:,1),rgb_mat(:,2),rgb_mat(:,3),'.');
% plot3(mean(rgb_mat(:,1)),mean(rgb_mat(:,2)),mean(rgb_mat(:,3)),'r.','markersize',10);
rgb_unsorted = rgb_mat_orig;
rgb_mat = sort(rgb_mat);
% min_perc= rgb_mat(N_trials*2.5/100,:);
% view(135,45);
% max_perc= rgb_mat(N_trials*97.5/100,:);


%fit a gaussian
M = [1/sqrt(2), -1/sqrt(2), 0 ; 1/sqrt(6) , 1/sqrt(6), -2/sqrt(6)]; %orthonormal basis for the simplex

y = (M*rgb_unsorted')';
mu = mean(y)';
sig = std(y)';
rho = (mean(prod(y,2))-prod(mu))/prod(sig);

Cov = [sig(1)^2 rho*prod(sig) ; rho*prod(sig) sig(2)^2];

g = @(x) exp(-1/2 *(M*x-mu)' * Cov^-1 * (M*x-mu)) / sqrt((2*pi)^2*det(Cov));

[X,Y] = meshgrid(linspace(-.1,.1,3e2),linspace(-.1,.1,3e2));
U = X(:)*M(1,:) + Y(:)*M(2,:) + repmat(mean(rgb_unsorted),9e4,1);
Z=[];
for ii = 1:size(U,1)
    Z(ii) = g(U(ii,:)');
end
U = reshape(U,3e2,3e2,3);
Z = reshape(Z,3e2,3e2);
% meshplot = mesh(U(:,:,1),U(:,:,2),U(:,:,3),Z);
% scatterplot.Visible='off';
% stamplot= plot3(2/9, 4/9, 3/9, 'g.','markersize',10)

conf_interval = diag(sqrt((M*eye(3))'*Cov * (M*eye(3))));
means = mean(rgb_unsorted)';

interval=[];
for jj = 1:3
    interval(1,jj)=mean(abs(rgb_unsorted(:,jj)-means(jj))<=2*conf_interval(jj)); %2 sigma correspond to 95%
end
means'
x_err=2*conf_interval'
interval
%% ternary plot
addpath(genpath('./MiRacleForPub/ternary/'));

%Create and plot the distribution of possible results according to
%statistical analysis. The plotting is based on 
%Ternary Plots file exchange functions by
% Ulrich Theune (2023). Ternary Plots (https://www.mathworks.com/matlabcentral/fileexchange/7210-ternary-plots). 
% MATLAB Central File Exchange. Retrieved September 4, 2023.
figure(3)
clf
clevels=4;

[h,hg,htick]=terplot(11);
c1=rgb_unsorted(:,1);
c2=rgb_unsorted(:,2);
X=0.5-c1*cos(pi/3)+c2/2;
Y=0.866-c1*sin(pi/3)-c2*cot(pi/6)/2;

Color=colororder({'#DE8F05','#0173B2','#029E73'});
M = [1/sqrt(2), -1/sqrt(2), 0 ; 1/sqrt(6) , 1/sqrt(6), -2/sqrt(6)]; %orthonormal basis for the simplex

y = (M*rgb_unsorted')';
mu = mean(y)';
sig = std(y)';
rho = (mean(prod(y,2))-prod(mu))/prod(sig);

Cov = [sig(1)^2 rho*prod(sig) ; rho*prod(sig) sig(2)^2];
g = @(x) exp(-1/2 *(M*x-mu)' * Cov^-1 * (M*x-mu)) / sqrt((2*pi)^2*det(Cov));


cldensity=zeros(size(rgb_unsorted,1),1);
clcontour=zeros(size(rgb_unsorted,1),1);
[N_bin,Xedges,Yedges,binX,binY] = histcounts2(y(:,1),y(:,2),30);
for i=1:size(rgb_unsorted,1)
    cldensity(i)=N_bin(binX(i),binY(i));
    clcontour(i)=g(rgb_unsorted(i,:)');
end

cmap253=colormap("jet");
hd=ternaryc(rgb_unsorted(:,1),rgb_unsorted(:,2),rgb_unsorted(:,3),cldensity);
hold on
N=1000;
[XX,YY] = meshgrid(linspace(min(X)-0.2, max(X)+0.2, N), linspace(min(Y)-0.2, max(Y)+0.2, N));
zi = griddata(X,Y,clcontour,XX,YY);
ct=contour(XX,YY,zi,clevels,'-w','linewidth',1.5);

c1=rgb111_mat(:,1);
c2=rgb111_mat(:,2);
X=0.5-c1*cos(pi/3)+c2/2;
Y=0.866-c1*sin(pi/3)-c2*cot(pi/6)/2;
y = (M*rgb111_mat')';
mu = mean(y)';
sig = std(y)';
rho = (mean(prod(y,2))-prod(mu))/prod(sig);

Cov = [sig(1)^2 rho*prod(sig) ; rho*prod(sig) sig(2)^2];

g = @(x) exp(-1/2 *(M*x-mu)' * Cov^-1 * (M*x-mu)) / sqrt((2*pi)^2*det(Cov));
cldensity111=zeros(size(rgb111_mat,1),1);
clcontour=zeros(size(rgb111_mat,1),1);

[N_bin,Xedges,Yedges,binX,binY] = histcounts2(y(:,1),y(:,2),30);
for i=1:size(rgb111_mat,1)
    cldensity111(i)=N_bin(binX(i),binY(i));
    clcontour(i)=g(rgb111_mat(i,:)');
end


N=1000;
[XX,YY] = meshgrid(linspace(min(X)-0.2, max(X)+0.2, N), linspace(min(Y)-0.2, max(Y)+0.2, N));
zi = griddata(X,Y,clcontour,XX,YY);

hd2=ternaryc(rgb111_mat(:,1),rgb111_mat(:,2),rgb111_mat(:,3),cldensity111);
colormap(cmap253);

hold on

ct111=contour(XX,YY,zi,clevels,'-w','linewidth',1.5);

hlabels=terlabel('miR-15b','miR-155','miR-126');
set(hg(:,1),'color',[Color(1,:),0.7])
set(hg(:,2),'color',[Color(2,:),0.7])
set(hg(:,3),'color',[Color(3,:),0.7])
set(hg,'linewidth',3,'lineStyle',':')

set(hlabels,'fontsize',20,'fontweight','bold')
set(hlabels(1),'color',Color(1,:))
set(hlabels(2),'color',Color(2,:))
set(hlabels(3),'color',Color(3,:))
%--  Modify the tick labels
set(htick,'fontsize',16,'fontweight','bold')

set(htick(:,1),'color',Color(1,:),'linewidth',3)
set(htick(:,2),'color',Color(2,:),'linewidth',3)
set(htick(:,3),'color',Color(3,:),'linewidth',3)
cb=colorbar;
cb.Limits=[1,max([cldensity,cldensity111],[],'all')+1];
cb.Ticks=round(linspace(1,max([cldensity,cldensity111],[],'all'),11));
cb.Label.String='Counts';
cb.FontSize=16;
cb.Label.FontSize=20;



