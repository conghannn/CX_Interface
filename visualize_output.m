%% This visualization code requires MATLAB 2014b or later
% at the end of execution, a movie named "testnat.mp4" will be created.
% The visualization here assumes that you use the default specification,
% i.e., running the simulation by leaving configuration file blank.

%% read simulation results
% read ommatidia coordinates
elevr1 = h5read('retina_elev.h5','/array');
azimr1 = h5read('retina_azim.h5','/array');
r1 = 1;
y1 = -r1 .* cos(elevr1) .* sin(azimr1);
x1 = -r1 .* cos(elevr1) .* cos(azimr1);
z1 = r1 .* sin(elevr1);

% read screen coordinates
elevr2 = h5read('grid_dima.h5','/array');
azimr2 = h5read('grid_dimb.h5','/array');
r2 = 10;
y = -r2 .* cos(elevr2) .* sin(azimr2);
x = -r2 .* cos(elevr2) .* cos(azimr2);
z = r2 .* sin(elevr2);

% read output file
data_BU = transpose(h5read('BU_output_pick.h5','/array'));
data_bu2 = transpose(h5read('bu2_output_pick.h5','/array'));
retina_out = transpose(h5read('retina_output0.h5','/spike_state/data'));
retina_array = transpose(h5read('retina_array.h5','/array'));

uids = transpose(h5read('retina_output0.h5','/spike_state/uids'));
temp = [];
for i = 1:721
    nid = regexp(cell2mat(uids(i)), 'neuron_R1_', 'split');
    num = str2num(nid{2}) + 1;
    temp = [temp,num];
end
[~, sort_index] = sort(temp);
retina_out = retina_out(:,sort_index);
screen = max(h5read('intensities.h5','/array'),0);

total_columnar = 14;

window_size = 25; %one side
spike_retina = zeros(2000, 721);
spike_BU = zeros(2000, 80);
spike_bu2 = zeros(2000, 80);

for i = 1:2000
    if i <= window_size
        temp_retina = sum(retina_out(1:i+window_size, :))/(i+window_size) * 2000;
        temp_BU = sum(data_BU(1:i+window_size, :))/(i+window_size) * 2000;
        temp_bu2 = sum(data_bu2(1:i+window_size, :))/(i+window_size) * 2000;
    elseif i > 2000 - window_size
        temp_retina = sum(retina_out(i-window_size:2000, :))/(2001 + window_size - 1) * 2000;
        temp_BU = sum(data_BU(i-window_size:2000, :))/(2001 + window_size - 1) * 2000;
        temp_bu2 = sum(data_bu2(i-window_size:2000, :))/(2001 + window_size - 1) * 2000;
    else
        temp_retina = sum(retina_out(i-window_size:i+window_size,:))/(2*window_size + 1) * 2000;
        temp_BU = sum(data_BU(i-window_size:i+window_size,:))/(2*window_size + 1) * 2000;
        temp_bu2 = sum(data_bu2(i-window_size:i+window_size,:))/(2*window_size + 1) * 2000;
    end

    spike_retina(i, :) = temp_retina;
    spike_BU(i, :)= temp_BU;
    spike_bu2(i, :) = temp_bu2;
end









%% visualization

% setup color axis
boxbg = [1,1,1]*0.9569;
weight = 14; % 10
screen_caxis = [0, max(max(max(screen)))];
screen_caxis_gc = [min(log10(screen(:))), max(log10(screen(:)))];
% input_caxis = [0, max(max(max(R1input)))];
% input_caxis_gc = [min(log10(R1input(:))), max(log10(R1input(:)))];


view1 = [-60, 18];
view2 = [180, 0];

cmap = gray(256);

% set up video writer

fig1 = figure();
set(fig1,'NextPlot','replacechildren');
% set(fig1, 'Color', [0.392, 0.475, 0.635])
set(fig1, 'Color', [1,1,1])

%set(fig1,'Position',[50,301, 640, 360]);
set(fig1,'Position',[50,301, 800, 600]);
winsize = get(fig1,'Position');
winsize(1:2) = [0 0];

write_to_file = 1;
if write_to_file
    writerObj = VideoWriter('testnat.mp4', 'MPEG-4');
    writerObj.FrameRate = 20;
    wtiterObj.Quality = 100;

    open(writerObj);
end
cmap = gray(256);

% start iterating through frames. The first 0.21 seconds are omitted
% view([0, 90])
for i = 1:size(screen,3)
    % screen intensity
    axis1 = subplot('position', [0.05, 0.6, 0.4, 0.35]);
    p1 = surf(x,y,z, screen(:,:,(i-1)/1+1), 'edgecolor','none');
    colormap('gray')
    view(view2)
    caxis(screen_caxis)
    axis equal
    xlim([-10,10])
    ylim([-10,10])
    shading interp
    grid off
    xlabel('x', 'FontSize', 16)
    ylabel('y', 'FontSize', 16)
    zlabel('z', 'FontSize', 16)
    set(axis1, 'Color', boxbg)
    title('Screen Intensity ', 'FontSize', 16, 'FontWeight', 'bold')

    index = i*10;
    axis2 = subplot('position', [0.5, 0.10, 0.4, 0.35]);
    a = zeros(8,10);
    for j = 1:80
        idy = mod(j-1,8) + 1;
        idx = 9 - floor((j-1)/8) + 1;
        a(idy,idx) = spike_BU(index, j);
    end

    h2 = surf(a, 'edgecolor','none');
    colormap('gray')
    view([0, 90])
    axis equal
    xlim([1,10])
    ylim([1,8])

    grid off
    xlabel('x', 'FontSize', 16)
    ylabel('y', 'FontSize', 16)
    zlabel('z', 'FontSize', 16)

    set(axis2, 'Color', boxbg)
    title('BU', 'FontSize', 16, 'FontWeight', 'bold')


    axis3 = subplot('position', [0.5, 0.6, 0.4, 0.35]);

    b = zeros(8,10);
    for j = 1:80
        idy = mod((j-1),8) + 1;
        idx = 9 - floor((j-1)/8) + 1;
        b(idy,idx) = spike_bu2(index, j);
    end
    g3 = surf(b, 'edgecolor','none');
    colormap('gray')
    view([0, 90])
    axis equal
    xlim([1,10])
    ylim([1,8])

    grid off
    xlabel('x', 'FontSize', 16)
    ylabel('y', 'FontSize', 16)
    zlabel('z', 'FontSize', 16)

    set(axis3, 'Color', boxbg)
    title('bu', 'FontSize', 16, 'FontWeight', 'bold')
    
    
    axis4 = subplot('position', [0.05, 0.10, 0.4, 0.4]);
    
    x_pick = [];
    y_pick = [];
    z_pick = [];
    for j = 1:721
        x_pick = [x_pick, 0 - retina_array(j, 1)];
        y_pick = [y_pick, retina_array(j, 2)];
        z_pick = [z_pick, spike_retina(index, j)];
    end
        
    scatter(x_pick,y_pick,25,z_pick, 'filled');
       
    axis equal
    xlim([-0.9,0.9])
    ylim([-0.9,0.9])
    
    grid off
    xlabel('x', 'FontSize', 16)
    ylabel('y', 'FontSize', 16)
    zlabel('z', 'FontSize', 16)

    set(axis4, 'Color', boxbg)
    title('retina0', 'FontSize', 16, 'FontWeight', 'bold')

    % Timer
anna1 = annotation('textbox', [0.40,0.99,0.1,0.01],...
    'String', [num2str((i)*1), 'ms'], ...
    'FontSize', 16, 'FontWeight', 'Bold', 'LineStyle','none');  

    drawnow
    if write_to_file
        frame = getframe(fig1,winsize);
        writeVideo(writerObj,frame);
    end

    pause(0.001)
%     delete(h2)
%     delete(h6);
%     delete(h5)
%     delete(h12);
    delete(anna1);
%     delete(anna3);
%     delete(anna3a);
end


if write_to_file
    close(writerObj);
end
