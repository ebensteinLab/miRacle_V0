% File to create PSF simulations that were used for the publication of
% MiRACLE method.

%% Input spectral information of fluorophores and filters
myFolder=fullfile("./MiRacleForPub","FluorophoresSpectra");
addpath(fullfile("./MiRacleForPub","spectral_color"));

fileEMPattern = fullfile(myFolder,'**', '*EM.txt'); % Change to whatever pattern you need.
fileEXPattern = fullfile(myFolder,'**', '*Abs.txt'); % Change to whatever pattern you need.

theEMFiles = dir(fileEMPattern);
theEXFiles=dir(fileEXPattern);

NamesToFind={'Alexa Fluor 647','Alexa Fluor 568','Alexa Fluor 546','Alexa Fluor 488 (H2O)'};
% NamesToFind={'ATTO 647N','ATTO 550','ATTO 532','ATTO 565','Alexa Fluor 488 (H2O)'};

EM=cell(1,length(NamesToFind));
EMname=strings;
EX=cell(1,length(NamesToFind));
EXname=strings;
nm=1;
for k = 1 : length(theEMFiles)  
    [~,name,~]=fileparts(theEMFiles(k).folder);
    if  sum(strcmp(name,NamesToFind),"all")
           fullFileName = fullfile(theEMFiles(k).folder, theEMFiles(k).name);
           EM{nm}=table2array(readtable(fullFileName));
           EMname(nm)=theEMFiles(k).name;
           fullFileName = fullfile(theEXFiles(k).folder, theEXFiles(k).name);
           EX{nm}=table2array(readtable(fullFileName));
           EXname(nm)=theEXFiles(k).name;
           nm=nm+1;
    end
end
 
 [~,EMnames,~]=fileparts(EMname);
 [~,EXnames,~]=fileparts(EXname);
 %% Dispersion calculation
 PixWLCalibration=importdata(fullfile("./MiRacle","PixWLCalibration.mat"));
 RPA=177.5;

RPAdispCurve=@(wl) 2*feval(PixWLCalibration,wl).*sind((180-RPA)/2);
wl_lim=[470 850];
invDispCurve=@(y) fzero(@(x) (RPAdispCurve(x)-y),wl_lim);
wl= EM{1}(:,1);
X_adj=RPAdispCurve(wl(wl>wl_lim(1)&wl<wl_lim(2)));
wl_adj=wl(wl>wl_lim(1)&wl<wl_lim(2));
%%
%
FluorLabel=EMnames;

EMadj=cellfun(@(x) x(x(:,1)>wl_lim(1)&x(:,1)<wl_lim(2),:),EM,'UniformOutput',false);
EXadj=cellfun(@(x) x(x(:,1)>wl_lim(1)&x(:,1)<wl_lim(2),:),EX,'UniformOutput',false);
SpectralOffset=-40;
LaserLines=[488,561,638];


maxLambdaFl=cellfun(@(x) max(x(x(:,2)==max(x(:,2)),1)), EMadj,'UniformOutput',true);
[maxLambdaFlSorted,indLambdaFl]=sort(maxLambdaFl);
EMsorted=EMadj(indLambdaFl);
EXsorted=EXadj(indLambdaFl);
FluorLabel=FluorLabel(indLambdaFl);
FluorLabel=cellfun(@(x) strrep(x,'Alexa Fluor ','AF'),FluorLabel,'UniformOutput',false);
FluorLabel=cellfun(@(x) erase(x,' - Em'),FluorLabel,'UniformOutput',false);
FluorLabel=cellfun(@(x) erase(x,' (H2O)'),FluorLabel,'UniformOutput',false);

sRGBFl=squeeze(spectrumRGB(maxLambdaFlSorted+SpectralOffset));

sRGBLasers=spectrumRGB(LaserLines);

Filters=FilterSpectrum(0);
FilterRange=[Filters(1,:)-Filters(2,:)./2;Filters(1,:)+Filters(2,:)./2];
FilterRange=FilterRange(:,FilterRange(1,:)>wl_lim(1));
tf=false(length(maxLambdaFlSorted),size(FilterRange,2));

dyeInFilter=zeros(length(maxLambdaFlSorted),size(FilterRange,2));
for dye=1:length(maxLambdaFlSorted)
for p=1:size(FilterRange,2)
    dyeInFilter(dye,p)=sum(EMsorted{dye}(wl_adj<min(wl_lim(2),FilterRange(2,p)) & wl_adj>max(wl_lim(1),FilterRange(1,p)),2));
end
end
[~,indFilt]=max(dyeInFilter,[],2);
Filters=Filters(:,indFilt');
x_ind_in_filter=any(x_ind_for_color,2);
FSadj=table(double(wl_adj),double(x_ind_in_filter));

PatchFiltersX=repelem(FilterRange(:,indFilt)',1,2);
PatchFiltersY=repmat([0,1,1,0],size(Filters,2),1);
tf_closeNames=diff(maxLambdaFlSorted)<30;
tf_ccloseNames=([false,diff(tf_closeNames)==0])&tf_closeNames==1;
y_forlabel=ones(1,length(maxLambdaFlSorted)).*0.85;
y_forlabel([false,tf_closeNames])=0.88;
y_forlabel([false,tf_ccloseNames])=0.82;

fig=figure(1);
clf
hSpectra=axes(fig);

hold on

AxesH = axes('Parent', fig, ...
  'Units', 'normalized', ...
  'Position', [hSpectra.Position(1), 0, hSpectra.Position(3), 1], ...
  'Visible', 'off', ...
  'XLim', [0, 1], ...
  'YLim', [0, 1], ...
  'NextPlot', 'add');
hold on
xlim(hSpectra,[min(X_adj) max(X_adj)])
hWl = copyobj(hSpectra,fig);
set(hWl,'Color','none')
set(hWl,'Ytick',[])
set(hWl,'XAxisLocation','top')
linkaxes([hSpectra,hWl,AxesH],'x')
x_ind_for_color=false(size(EMsorted{1},1),length(EMsorted));
for i=1:length(EMsorted)
h_ex(i)=plot(hSpectra,X_adj,EXsorted{i}(:,2),'LineStyle',':','color',sRGBFl(i,:),'LineWidth',2);
x_ind_for_color(:,i)=EMsorted{i}(:,1)>=PatchFiltersX(i,1)&EMsorted{i}(:,1)<=PatchFiltersX(i,3);

h_gray(i)=area(hSpectra,X_adj,EMsorted{i}(:,2),'FaceColor',[.5 .5 .5],'FaceAlpha',0.3,'LineStyle','none');
h_em(i)=area(hSpectra,X_adj(x_ind_for_color(:,i),1),EMsorted{i}(x_ind_for_color(:,i),2),'FaceColor',sRGBFl(i,:),'FaceAlpha',0.5,'LineStyle','none');

text(AxesH,RPAdispCurve(maxLambdaFlSorted(i)),y_forlabel(i), FluorLabel{i},'HorizontalAlignment','center', 'BackgroundColor', 'none','FontSize',24,'Rotation',0);
end
for i=1:length(LaserLines)
    hx(i)=xline(hSpectra,RPAdispCurve(LaserLines(i)),'color',sRGBLasers(1,i,:),'linewidth',5);
    text(hSpectra,RPAdispCurve(LaserLines(i)-6),0.5, [num2str(LaserLines(i)),' nm'],'HorizontalAlignment','center', 'BackgroundColor', 'none','FontSize',24,'Rotation',90);

end
for i=1:size(PatchFiltersX,1)

    hp(i)=patch(hSpectra,RPAdispCurve(PatchFiltersX(i,:)),PatchFiltersY(i,:),sRGBFl(i,:),'faceAlpha',0.05,'linestyle',':','edgeColor',sRGBFl(i,:));

end

set(hSpectra,...
    'FontSize',24,...
    'Box','on',...
    'LineWidth',2,...
    'BoxStyle','full',...
    'YTick',[]);
ylim(hSpectra,[0 1])
xlabel(hSpectra,'Pixel Displacement [pix]');



Pix_xticks=get(hWl,'XTick');
for i=1:length(Pix_xticks)
    converted_values(i)=invDispCurve(Pix_xticks(i));
end
set(hWl,'XTickLabel',round(converted_values),'FontSize',24)

xlabel(hWl,'Wavelength [nm]')
set(hSpectra,'Position',hWl.Position)

%% Plot PSFs simulation
%miR1= 488-647 [1,4]
%miR2= 546-647 [2,4]
%miR3= 488-568 [1,3]


%To make a more realistic intensity distribution between the two fluorophores
% It is possible to introduce QY and EX [AF488 AF546 AF568 AF647] taken from  ...
%https://www.bu.edu/flow-cytometry/files/2013/06/Fluorochromes-Brightness-Chart.pdf
% and FRET calculation based on the dyes R0 FRET distance. This does not
% give a significant change, therefor we did not use this in our
% publication, however it can be commented in to provide a better simulation. 
QY_BD=[.94 .96 .75 .33]; 
Ex_BD=[71000 104000 91300 239000];
Brightness_BD=QY_BD.*Ex_BD/max(QY_BD.*Ex_BD);

QY_AAT=[.92 .79 .69 .33];
Ex_AAT=[73000 112000 88000 270000];
Brightness_AAT=QY_AAT.*Ex_AAT/max(QY_AAT.*Ex_AAT);

bp_dist=23;
stretch_factor=.95; % additional parameter due to contact with the S9.6 antibody
r=bp_dist*0.33*stretch_factor; %distance between dyes in nm
R0=[5.568 7.101 6.074]; %Förster Radius in nm from https://www.fpbase.org/fret/
R0QYa=[1.838 2.343 4.191];
FRETeff= 1./(1+(r./R0).^6);

SNR=10;
m=.3;
sigm=1.15;

mirNames={'15b-5p','155-5p','126-3p'};

invDispCurve=@(y) fzero(@(x) (RPAdispCurve(x)-y),wl_lim);
wl= EM{1}(:,1);
pix_disp=RPAdispCurve(wl);
X_adj=RPAdispCurve(wl(wl>wl_lim(1)&wl<wl_lim(2)));

color_pair= [1,4
            2,4
            1,3];
figure(2)
clf
T=tiledlayout(3,1,"TileSpacing","compact","Padding","compact");

for i=1:size(color_pair,1)
    t=tiledlayout(T,1,3);
    t.Layout.Tile=i;
    nexttile(t);
    color_seq=color_pair(i,:);
lasers=[488,561,561,640];
exfactor=zeros(1,length(EMnames));
for ex=1:length(EXsorted)
    exfactor(ex)= EXsorted{ex}(EXsorted{ex}(:,1)==lasers(ex),2);
end
num_locs=2;
sz=25;
x_displace=0;
sim_loc=zeros(sz);
locs=zeros(num_locs,2)+ceil(sz/2);
locs(:,2)=locs(:,2)-7;
x_ind_in_filter=any(x_ind_for_color,2);
gaussfun=@(x,y,a) a(3)*exp(-((x-a(1)).^2+(y-a(2)).^2)/(2*a(4)^2));
im_sz=size(sim_loc,1,2);
[Xpos,Ypos]=meshgrid(1:im_sz(2),1:im_sz(1));

for loc=1:size(locs,1)
    simParam=[locs(loc,1),locs(loc,2),1,sigm];
    sim_loc=sim_loc+gaussfun(Xpos,Ypos,simParam);
end


imagesc(sim_loc)
set(gca,'XTick',[],'YTick',[])
colormap gray

title('No Dispersion','fontsize',16,'fontweight','bold')
sim_disp=zeros(sz);

for cl=1:length(color_seq)
color=color_seq(cl);

ParamSpectral=[repmat(locs(cl,1),length(X_adj(:)),1),repmat(locs(cl,2),length(X_adj(:)),1)+X_adj(:),FSadj.Var2.*EMsorted{color}(:,2),repmat(sigm,length(X_adj(:)),1)];
ParamSpectral(:,3)=ParamSpectral(:,3)*exfactor(color)./sum(EMsorted{color}(:,2));%*Brightness_BD(color)*(1+sign(cl-1.5)*FRETeff(i))
for spec=1:size(ParamSpectral)
    sim_disp=sim_disp+gaussfun(Xpos,Ypos,ParamSpectral(spec,:));
end

end
sim_disp=sim_disp./max(sim_disp,[],'all');

Gauss_Var=(1/SNR)^2;
noisy_sim_disp=imnoise(sim_disp,'poisson');
noisy_sim_disp=(imnoise(noisy_sim_disp,'gaussian',m,Gauss_Var));

[pks,locs]=findpeaks(double(noisy_sim_disp(:,ceil(sz/2))),'NPeaks',2,'SortStr','descend');
locs=sort(locs);
pair_dist=diff(locs);

axis image
nexttile
imagesc(sim_disp)
colormap gray
hold on
scatter(repmat(ceil(sz/2),1,2),locs,25,sRGBFl(fliplr(color_seq),:),'filled');

set(gca,'XTick',[],'YTick',[])
title('Dispersed','fontsize',16,'fontweight','bold')
title(t,[['miR barcodes simulation RPA ',num2str(RPA)],['miR',int2str(i),'-',mirNames{i},': ',FluorLabel{color_seq(1)},'-',FluorLabel{color_seq(2)}],""],'fontsize',20,'fontweight','bold')
xlabel(['Distance between peaks = ',num2str(pair_dist),' pixels'])
axis image

nexttile
imagesc(noisy_sim_disp)
colormap gray

set(gca,'XTick',[],'YTick',[])
title('Dispersed noisy','fontsize',16,'fontweight','bold')
xlabel(['gaussian VAR = ',num2str(Gauss_Var)])
colorbar
axis image
end