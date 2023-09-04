load('./MiRacleForPub/HistAndMatData300423.mat')

DistHistData=HIstogramData(2:end,:);
DistHistData{:,:}=HIstogramData{2:end,:}./sum(HIstogramData{2:end,:},1);
Normalized253HistData=DistHistData(:,1:2);
tempNorm253HistData=DistHistData{:,1:2}./DistHistData.mix111;
Normalized253HistData{:,:}=tempNorm253HistData./sum(tempNorm253HistData,1);

 %% normalized 253 histogram
% figure(1)
% clf
% label_names=Normalized253HistData.Properties.RowNames;
% 
% catlabels253=reordercats(categorical(label_names),label_names);
% 
% bar(catlabels253,Normalized253HistData{:,:})
% hold on
% bar(categorical(Normalized253HistData.Properties.RowNames),[2,5,3]./sum([2,5,3]),'FaceColor','none','LineWidth',2,'LineStyle',':','EdgeColor',[.4 .4 .4],'BarWidth',0.7)
% ylim([0 0.52])
% hold off
% % legend([Normalized253HistData.Properties.VariableNames,{'Expected ratio'}])
% legend({'All', 'Visually labeled' ,'Expected ratio'})
% set(gca,'FontSize',20)
% ylabel('Fraction')
% title('Synthetic MiR Mixture 2:5:3')
% xlim tight
% 

%% confusion charts
label_names=LabeledValidationSet.Properties.RowNames;
label_names{1}='Noise';
LabeledValidationSet.Properties.RowNames=label_names;
LabeledValidationSet.Properties.VariableNames=label_names;
LabeledTestSet.Properties.RowNames=label_names;
LabeledTestSet.Properties.VariableNames=label_names;
Labeled253Set.Properties.RowNames=label_names;
Labeled253Set.Properties.VariableNames=label_names;
singleChartFlag=1;
figure(2)
clf
if singleChartFlag
cmTest=confusionchart(LabeledTestSet{:,:},LabeledTestSet.Properties.RowNames);

cmTest.ColumnSummary='column-normalized';
cmTest.RowSummary='row-normalized';

cmTest.XLabel='';
cmTest.YLabel='';

sortClasses(cmTest,label_names);
set(gca,'FontSize',20)

else
tiledlayout(2,1)
nexttile
cmVal=confusionchart(LabeledValidationSet{:,:},LabeledValidationSet.Properties.RowNames);
cmVal.ColumnSummary='column-normalized';
cmVal.RowSummary='row-normalized';
cmVal.Title=["Validation set" ,"(10% of visully-labeled single-species samples)"];
cmVal.Normalization='absolute';
set(gca,'FontSize',16)
sortClasses(cmVal,label_names);
cmVal.XLabel='Classifier prediction';
cmVal.YLabel='Visually labeled';


nexttile
cmMix=confusionchart(Labeled253Set{:,:},Labeled253Set.Properties.RowNames);
cmMix.ColumnSummary='column-normalized';
cmMix.RowSummary='row-normalized';
cmMix.Title=["Test set", "(visually labeled 2:5:3 mixture)"];
cmMix.Normalization='absolute';
sortClasses(cmMix,label_names);
cmMix.XLabel='Classifier prediction';
cmMix.YLabel='Visually labeled';
set(gca,'FontSize',16)
end
%% Unmixed Histograms
figure(3)
T=tiledlayout(3,1);
UnmixHist=HIstogramData(:,5:end);
DistUnmixHist=UnmixHist;
DistUnmixHist{:,:}=UnmixHist{:,:}./sum(UnmixHist{:,:},1);

C=colororder({'#0173B2','#DE8F05','#029E73'});
for col=1:size(DistUnmixHist,2)
b(col)=nexttile;
catlabels=reordercats(categorical(label_names),label_names);
bar(catlabels(2:end),UnmixHist{2:end,col},'FaceColor',C(col,:));
hold on
bar(catlabels(1),UnmixHist{1,col},'FaceColor',C(col,:),'FaceAlpha',.1,'LineStyle','--');
hold off
ylabel('Counts')
set(gca,'FontSize',16)

legend(label_names(col+1))
xlim tight
end

title(T,'Single miR type samples classification','fontSize',20,'FontWeight','bold');
%% PR curves
figure(4)
clf
C=colororder({'#0173B2','#DE8F05','#029E73'});
pr_labels=string(label_names(2:end))+' pp';
argmax_labels=string(label_names(2:end))+' argmax';
fig=figure(11);
cmTest=confusionchart(LabeledTestSet{:,:},LabeledTestSet.Properties.RowNames);
figure(4)
cmTest.Normalization='column-normalized';
Precision=diag(cmTest.NormalizedValues);
cmTest.Normalization='row-normalized';
Recall=diag(cmTest.NormalizedValues);
cmTest.Normalization='absolute';
cmTest.sortClasses(label_names);
ConfMat=cmTest.NormalizedValues;
pnValues=sum(ConfMat,2);
hold on
for mir=1:3
pr(mir)=plot(PR_data{:,2*mir-1},PR_data{:,2*mir},'LineWidth',4,'Color',C(mir,:));
sc(mir)=scatter(Recall(mir+1),Precision(mir+1),200,C(mir,:),"filled");
bl(mir)=yline(pnValues(mir+1)/pnValues(1),'Color',C(mir,:),'LineStyle','--','LineWidth',4);

end
AP=[0.91 0.75 0.86];
legend_entries=[label_names(2:end)+" ("+string(AP)'+")"];
legend([pr],[legend_entries ],'Location','west','FontSize',20)
AP=[0.91 0.75 0.86];
ylabel('Precision')
xlabel('Recall')
set(gca,'FontSize',24,'XGrid',1,'Ygrid',1,'XMinorGrid',1,'YMinorGrid',1)
title(' 1:1:1 Mixture Classification','FontSize',28)

hold off
close(fig)
%% mixed Histograms
figure(5)
clf
% T=tiledlayout(2,1);
% nexttile
mixHist=HIstogramData(:,[1:4]);
DistmixHist=mixHist(2:4,:);
DistmixHist{:,:}=mixHist{2:4,:}./sum(mixHist{2:4,:},1);
mixlabels=DistmixHist.Properties.VariableNames;
catlabels=categorical(DistmixHist.Properties.RowNames);
% 

barplotDataClassifier=zeros(3,4);
barplotDataVisual=zeros(3,4);
barplotDataClassifier(:,[1,3])=mixHist{2:4,[3,1]};
barplotDataVisual(:,[2,4])=mixHist{2:4,[4,2]};
C111=C(1,:);
C253=C(2,:);

% yyaxis right
h1=bar(catlabels,barplotDataClassifier);
h1(1).FaceColor=C111(1,:);
h1(3).FaceColor=C253(1,:);
ylabel('Classifier counts','Color','k','FontSize',24)

ylim([0 max(barplotDataClassifier,[],'all')+0.1*max(barplotDataClassifier,[],'all')]);

yyaxis right

h2=bar(catlabels,barplotDataVisual);
h2(2).FaceColor=C111(1,:);
h2(2).FaceAlpha=.5;
h2(4).FaceColor=C253(1,:);
h2(4).FaceAlpha=.5;
set(gca,'FontSize',20)

legend([h1(1),h2(2),h1(3),h2(4)],{'Classifier prediction 1:1:1','Visually labeled 1:1:1',...
    'Classifier prediction 2:5:3','Visually labeled 2:5:3'},'Location','north','fontsize',20)
ylabel('Visually labeled counts','Color','k','FontSize',24)
ax=gca;
ylim([0 max(barplotDataVisual,[],'all')+0.1*max(barplotDataVisual,[],'all')]);
ax.YAxis(1).Color = 'k';
ax.YAxis(2).Color = 'k';
ax.YAxis(1).FontSize=24;
ax.YAxis(2).FontSize=24;
 ax.XAxis.FontSize=24;
 xlim tight

title(T,'Mixed samples classification','fontSize',20,'FontWeight','bold');
%% Fractrion histograms
figure(6)
clf
h1=bar(catlabels,DistmixHist{:,[3,4,1,2]});

h1(1).FaceColor=C111(1,:);
h1(2).FaceColor=C111(1,:);
h1(2).FaceAlpha=.5;

h1(3).FaceColor=C253(1,:);
h1(4).FaceColor=C253(1,:);
h1(4).FaceAlpha=.5;
set(gca,'FontSize',20)
legend([h1(1),h1(2),h1(3),h1(4)], ...
    {['Classifier prediction 1:1:1 (',num2str(sum(mixHist{2:4,"mix111"})),' crops)'],...
    ['Visually labeled 1:1:1 (',num2str(sum(mixHist{2:4,"Tinder111"})),' crops)'],...
    ['Classifier prediction 2:5:3 (',num2str(sum(mixHist{2:4,"mix253"})),' crops)'], ...
    ['Visually labeled 2:5:3 (',num2str(sum(mixHist{2:4,"Tinder253"})),' crops)']}, ...
    'Location','north','fontsize',20)
ylabel('Fraction','Color','k','FontSize',24)
ylim([0 max(DistmixHist{:,[3,4,1,2]},[],'all')+0.1*max(DistmixHist{:,[3,4,1,2]},[],'all')]);
ax=gca;
ax.YAxis.FontSize=24;
 ax.XAxis.FontSize=24;

%% Expected vs measured ratio 253 line
figure(7)
clf
label_names=Normalized253HistData.Properties.RowNames;
C=colororder({'#0173B2','#DE8F05','#029E73'});

catlabels253=reordercats(categorical(label_names),label_names);
hold on
line([0 0.52],[0 0.52],'LineStyle','--','Color',[0 0 0])

expectedRatio=[2,5,3]./sum([2,5,3]);
splot(1)=scatter(Normalized253HistData{:,1},expectedRatio,100,C,'filled','o');
splot(2)=scatter(Normalized253HistData{:,2},expectedRatio,100,C,'diamond');
text(mean(Normalized253HistData{:,:},2),expectedRatio-.025,label_names,'FontSize',14,'HorizontalAlignment','center')
hold off
% legend([Normalized253HistData.Properties.VariableNames,{'Expected ratio'}])
set(gca,'FontSize',20,'XGrid',1,'YGrid',1,'XMinorGrid',1,'YMinorGrid',1)
ylabel('Expected ratio')
xlabel('Measured ratio')
title(' 2:5:3 Mixture Quantification')
axis([0 0.52 0 0.52]) 
legend(splot,{'All data', 'Visually labeled'},'Location','northwest')
%% unconfusion matrix analysis
fig=figure(2);
clf
cmTest=confusionchart(LabeledTestSet{:,:},LabeledTestSet.Properties.RowNames);
cmTest.Normalization='row-normalized';
confMatNorm=cmTest.NormalizedValues;
mix253prediction=HIstogramData.mix253;
hatHistogramData=table('Size',size(HIstogramData),'VariableTypes',repmat("double",size(HIstogramData,2),1),'VariableNames',HIstogramData.Properties.VariableNames);
hatHistogramData{:,:}=(HIstogramData{:,:}'/confMatNorm)';
hat253values=mix253prediction'/confMatNorm;
normhat253= hat253values(2:end)./sum(hat253values(2:end));
label_names=DistHistData.Properties.RowNames;


hatDistHistData=hatHistogramData(2:end,:);
hatDistHistData{:,:}=hatHistogramData{2:end,:}./sum(hatHistogramData{2:end,:},1);
hatNormalized253HistData=hatDistHistData(:,1:2);
hattempNorm253HistData=hatDistHistData{:,1:2}./hatDistHistData.mix111;
hatNormalized253HistData{:,:}=hattempNorm253HistData./sum(hattempNorm253HistData,1);
Normalized253HistData;

catlabels253=reordercats(categorical(label_names),label_names);
figure(8)
bar(catlabels253,hatNormalized253HistData{:,:})
hold on
bar(categorical(Normalized253HistData.Properties.RowNames),[2,5,3]./sum([2,5,3]),'FaceColor','none','LineWidth',2,'LineStyle',':','EdgeColor',[.4 .4 .4],'BarWidth',0.7)
ylim([0 0.52])
hold off
legend({'All', 'Visually labeled' ,'Expected ratio'})
set(gca,'FontSize',24)
ylabel('Fraction')
title(' 2:5:3 Mixture Quantification')
xlim tight

figure(9)
x_err=   [0.0444    0.0786    0.0607];
plot_lim=[.1 0.6];
clf
label_names=Normalized253HistData.Properties.RowNames;
C=colororder({'#0173B2','#DE8F05','#029E73'});

catlabels253=reordercats(categorical(label_names),label_names);
hold on
line([plot_lim(1) plot_lim(2)],[plot_lim(1) plot_lim(2)],'LineStyle','--','Color',[0 0 0])

expectedRatio=[2,5,3]./sum([2,5,3]);
splot(1)=scatter(expectedRatio,hatNormalized253HistData{:,1},150,C,'filled','o');

splot(2)=scatter(expectedRatio,hatNormalized253HistData{:,2},100,C,'diamond','LineWidth',2);

for i=1:3
errorbar(expectedRatio(i),hatNormalized253HistData{i,1},x_err(i)/2,'vertical','LineStyle','none','Color',C(i,:),'LineWidth',2);
end
text(expectedRatio+.01,hatNormalized253HistData{:,1},label_names,'FontSize',20,'HorizontalAlignment','left')
hold off
set(gca,'FontSize',24,'XGrid',1,'Ygrid',1,'XMinorGrid',1,'YMinorGrid',1)
xlabel('Expected ratio')
ylabel('Measured ratio')
title(' 2:5:3 Mixture Quantification','FontSize',28)
axis([plot_lim(1) plot_lim(2) plot_lim(1) plot_lim(2)]) 
legend(splot,{'All data', 'Visually labeled'},'Location','northwest')
close(fig)

%% PCA Eigenfaces plot

path= '/Users/jonjeffet/Dropbox/My Mac (Jonathan’s MacBook Pro)/Downloads/eigenfaces/';
oldpath=cd(path);
filenames=dir('*.csv');
cd(oldpath);
eigenfaces=zeros(24,10,length(filenames));
for i=1:length(filenames)
file=fullfile(path,filenames(i).name);
eigenfaces(:,:,i)=readmatrix(file);
end
figure(11)
T=tiledlayout(2,10);
cl=zeros(20,2);
clear ax;

for i=1:size(eigenfaces,3)
nexttile
imagesc(eigenfaces(:,:,i));
ax(i)=gca;
    ax(i).Box='on';
    ax(i).LineWidth=2;
    ax(i).XTickLabel={""};
    ax(i).XTick=[];


title(num2str(i),'fontSize',20);
cl(i,:)=caxis;
colormap(flipud(lbmap(256,'BrownBlue')))
if i==10||i==20
colorbar(ax(i),'lineWidth',2) 
else
    colorbar('off')
end

end
for i=1:length(ax)
caxis(ax(i),[-.3,.3]);
ax(i).FontSize=20;

if i==1||i==11
    ax(i).Box='on';


else


ax(i).YTickLabel={""};

end
if i>=1&&i<=10
ax(i).XTickLabel={""};

end
end
title(T,'First 20 PCA Components','fontsize',26,'fontweight','bold');
T.TileSpacing='compact';
T.Padding='compact';


%% Augmented denoised crops

path= '/Users/jonjeffet/Dropbox/My Mac (Jonathan’s MacBook Pro)/Downloads/mirtype13is15b155126ofcourse/';
oldpath=cd(path);
filenames=dir('*.csv');
cd(oldpath);
crops=zeros(24,10,length(filenames));
for i=1:length(filenames)
file=fullfile(path,filenames(i).name);
crops(:,:,i)=readmatrix(file);
end
figure(12)
T=tiledlayout(3,2);
cl=zeros(6,2);
clear ax
for i=1:size(crops,3)
nexttile
imagesc(crops(:,:,i));
colormap("gray")
ax(i)=gca;

ax(i).Box='on';
ax(i).LineWidth=2;
ax(i).XTickLabel={""};
ax(i).XTick=[];
if i==1
    ylabel('miR-15b');
elseif i==3
    ylabel('miR-155');
elseif i==5
    ylabel('miR-126');

end

if mod(i,2)
title('Denoised','fontSize',20);

else
title('Augmented','fontSize',20);
end
cl(i,:)=caxis;
axis image

end
for i=1:length(ax)
caxis(ax(i),[min(cl,[],'all')+min(cl,[],'all')*0.01,max(cl,[],'all')-max(cl,[],'all')*0.1]);
ax(i).FontSize=20;


end
title(T,'Augmentation','fontsize',26,'fontweight','bold');
T.TileSpacing='compact';
T.Padding='compact';


