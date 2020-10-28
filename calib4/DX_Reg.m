%% Import & Parse Data
filename = '/Volumes/HENRY_TENG/Research/Calibrations/DX/Calibration_DX.csv';
delimiter = ',';
startRow = 9;
formatSpec = '%q%q%q%q%q%q%q%q%q%[^\n\r]';
fileID = fopen(filename,'r','n','UTF-8');
fseek(fileID, 3, 'bof');
temp = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
fclose(fileID);
temp = [temp{:}];
[m,n] = size(temp);
for i = 1:m
    for j = 1:n-1
        dat(i,j) = str2num(temp(i,j));
    end
end
        
img = dat(:,2);
DX = unique(dat(:,1));
deg = dat(:,3);
novo = dat(:,4:7);
p_X = dat(:,8);
p_Y = dat(:,9);

%% Generate Values
DZ = abs(mean(novo,2));
for i = 1:length(DX)
    ind = find(dat(:,1) == DX(i));
    b = polyfit(DZ(ind),p_X(ind),1);
    temp = [min(DZ(ind)):0.01:max(DZ(ind))];
    figure(i)
    plot(DZ(ind),p_X(ind),'b-o'), hold on
    plot(temp,b(2) + b(1).*temp, '--r')
    xlabel('DZ [in]')
    ylabel('[Pixel]')
    annotation('textbox',[0.2,0.5,0.3,0.3],'String',strcat('Slope: ',num2str(b(1))),'FitBoxToText','on');
    legend('p_X', 'Regression');
    title(strcat('DX: ', num2str(DX(i)), ' [m]'));
end
    

    
    



