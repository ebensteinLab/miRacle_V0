function Filters=FilterSpectrum(plot_flag)
%inputs: plot_flag: 1 for plotting (defaults to plotting)
%outputs: Spectral windows of the combined multiband filter and polychroic
%mirror. output as [2,nWindows] matrix, where the first row corresponds
%with the center wavelength of the window, and the second row is the width
%of transmission>0.4 of the window.
% Written by Jonathan Jeffet 

path=fullfile(pwd,"FilterSpectra");
if nargin==0
    plot_flag=1;
end

EmFilter=importdata(fullfile(path,'old MBF.txt'));
MBM5c=importdata(fullfile(path,'zt405_488_561_640_785 MBM.txt'));

CombinedSpectrum=MBM5c(MBM5c(:,1)>350&MBM5c(:,1)<1000,:);

CombinedSpectrum(:,2)=CombinedSpectrum(:,2).*EmFilter.data(ismember(EmFilter.data(:,1),CombinedSpectrum(:,1)),2);
if plot_flag
figure()
hold on
plot(EmFilter.data(:,1),EmFilter.data(:,2),'-b','LineWidth',2);
plot(MBM5c(:,1),MBM5c(:,2),'-g','LineWidth',2);
plot(CombinedSpectrum(:,1),CombinedSpectrum(:,2),'-k','LineWidth',2);
xlim([350 1000])
ylim([0 1])
end
[BWmap,n]=bwlabeln(CombinedSpectrum(:,2)>0.4);
SpectralWindow=zeros(n,2);
Filters=zeros(2,n);
for i=1:n
SpectralWindow(i,1)=CombinedSpectrum(find(BWmap(:)==i,1,'first'),1);
SpectralWindow(i,2)=CombinedSpectrum(find(BWmap==i,1,'last'),1);
Filters(1,i)=mean(SpectralWindow(i,:));
Filters(2,i)=diff(SpectralWindow(i,:));

end
end