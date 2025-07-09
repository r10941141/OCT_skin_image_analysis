function varargout = modification_gui(varargin)
% MODIFICATION_GUI MATLAB code for modification_gui.fig
%      MODIFICATION_GUI, by itself, creates a new MODIFICATION_GUI or raises the existing
%      singleton*.
%
%      H = MODIFICATION_GUI returns the handle to a new MODIFICATION_GUI or the handle to
%      the existing singleton*.
%
%      MODIFICATION_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MODIFICATION_GUI.M with the given input arguments.
%
%      MODIFICATION_GUI('Property','Value',...) creates a new MODIFICATION_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before modification_gui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to modification_gui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help modification_gui

% Last Modified by GUIDE v2.5 23-Oct-2022 23:37:12

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @modification_gui_OpeningFcn, ...
                   'gui_OutputFcn',  @modification_gui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT





% --- Executes just before modification_gui is made visible.
function modification_gui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to modification_gui (see VARARGIN)
%% processing / saving setting
global lines DyRange MaxImg MinImg flag4linear2save background_line x set_x hostfolder binBase fname2 fname3 Gy statu c bon
lines=944;
statu=0;
DyRange = 30; % setting dynamic range of the log scale OCT image
MaxImg = 5e5;
MinImg = 0;
bon = 0;
flag4linear2save = 1;
c = uicontrol('Style','radiobutton','String','Toggle Button','Position',[217 16 16 7],'Callback',@ttt);
%% automatic batch processing
background_line = 0; %need to adjust %background setting
x=1:1:944;
set_x=1:1:944;
hostfolder=uigetdir('.\auto_classification\');
%hostfolder =[participants '\wrong'];
binBase = dir([hostfolder '\OAC_image_size1024\*.pgm']);
%for binBaseInd=1:length(binBase(:));
cla;
if ~exist('binBaseInd','var') || isempty(handles.binBaseInd), handles.binBaseInd=1;  
end

binBaseInd=handles.binBaseInd;
fname = binBase(binBaseInd).name;
underscore_inds = strfind(fname, '.');
fname2=fname(1:underscore_inds-5);
n1=str2num(fname(underscore_inds-4:underscore_inds-1));
n1=n1/4;
fname3=[fname2 num2str(n1,'%04d')];

epidermis_struct=load([hostfolder '\epidermis_mat\' fname3 '.mat']);
OAC_1024image=imread([hostfolder '\OAC_image_size1024\' fname]);
handles.OAC=OAC_1024image;

%denoise
[thr,sorh,keepapp] = ddencmp('den','wv',OAC_1024image);
ixc=wdencmp('gbl',OAC_1024image,'sym4',5,thr,sorh,keepapp);
k3=medfilt2(ixc,[15 15]);
isuo=imresize(k3,1, 'bicubic' );
%
[Gx, Gy] = imgradientxy(isuo,'sobel');
X=zeros(1,lines); 
for x=1:1:lines
    [M,N]=max(Gy(10:200,x)); %set the boundary
    X(1,x)=N+10+bon;
end
%平滑度處理
for x=6:1:lines-10
    if abs(X(:,x-1)-X(:,x))>10
        X(:,x)=(X(:,x-3)+X(:,x+3))/2;
    elseif abs(X(:,x-3)-X(:,x))>20
        X(:,x)=(X(:,x-5)+X(:,x+5))/2;
    else
        X(:,x)=X(:,x);
    end
end
%
for x=1:1:50 %for bug of edges
    if abs(X(:,x)-X(:,x+10))>30
        X(:,x)=X(:,x+20);
    else
        X(:,x)=X(:,x);
    end
end

%%
[row,col,v]=find(epidermis_struct.epidermis_mat2);
count_col=zeros(1,1000);
y_2_surface=zeros(1,1000);
y_2_EDJ=zeros(1,1000);
for i=1:1:1000
    count_col(1,i)=sum(col(:)==i);

%     if i>1
        y_2_surface(i)=row(sum(count_col(1,1:i-1))+1);
        y_2_EDJ(i)=row(sum(count_col(1,1:i)));
%     else
%         y_2_surface(i)=row(1);
%         y_2_EDJ(i)=row(sum(count_col(1,1)));
%     end

end
y_surface=y_2_surface(1,30:end-27);
y_EDJ=y_2_EDJ(1,30:end-27);
balance_y=round(mean(X-y_surface));
Y_surface=y_surface+balance_y;
Y_EDJ=y_EDJ+balance_y;
newChr2 = extractAfter(fname3,strlength(fname3)-4);
handles.Y_surface=Y_surface;
handles.Y_EDJ=Y_EDJ;
axes(handles.axes1);
imagesc(handles.OAC),colormap(gray),hold on;
plot(handles.Y_surface,'r','LineWidth',1.2),hold on;
plot(handles.Y_EDJ,'b','LineWidth',1.2),hold on;
title('manual modification #'+string(newChr2),'FontSize',22);
set(gca,'YDir','reverse'),axis([0 944 0 370]);


% Choose default command line output for modification_gui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles); 

% This sets up the initial plot - only do when we are invisible
% so window can get raised using modification_gui.

% UIWAIT makes modification_gui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = modification_gui_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles) %update
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

cla;
statu=0;
global hostfolder fname3 lines
if isfield(handles, 'Y_surface2')==1
   handles.Y_surface=handles.Y_surface2;
end
if isfield(handles, 'Y_EDJ2')==1
   handles.Y_EDJ=handles.Y_EDJ2;
end
axes(handles.axes1);
imagesc(handles.OAC),colormap(gray),hold on;
drawnow;
plot(handles.Y_surface,'r','LineWidth',1.5),hold on;
plot(handles.Y_EDJ,'b','LineWidth',1.5),hold on;
set(gca,'YDir','reverse'),axis([0 944 0 370]),title('update result' ),xlabel('line (pixel)'),ylabel('depth (pixel)');


f=figure('visible','off');hold on;
imagesc(handles.OAC),colormap(gray),hold on;
plot(handles.Y_surface,'r','LineWidth',1.5),hold on;
plot(handles.Y_EDJ,'b','LineWidth',1.5),hold on;
set(gca,'YDir','reverse'),axis([0 944 0 370]),set(gca,'Position',[0,0,1,0.6]),title('update result' ),xlabel('line (pixel)'),ylabel('depth (pixel)');

rootFolder4modification_plot = [hostfolder '\modification_plot']; 
ImName_plot= [hostfolder '\modification_plot\' num2str(fname3)];
if exist(ImName_plot)==0
    mkdir(rootFolder4modification_plot);
end
saveas(f,ImName_plot,'png');

Y_surface=handles.Y_surface;
Y_EDJ=handles.Y_EDJ;
%save epidermis curve or line(use curve now)
ep_begin_position=round(min(Y_surface(1,:)));
dm_begin_position=round(min(Y_EDJ(1,:)));
ep_depth=(round(max(Y_EDJ(1,:)))-round(min(Y_surface(1,:))))+1;
epidermis_mask=zeros(1000,944);
for line3=1:1:lines
   epidermis_mask(round(Y_surface(1,line3)):(round(Y_EDJ(1,line3))),line3)=255;
end
epidermis_mask=uint8(epidermis_mask);
epidermis_mask=epidermis_mask(1:370,:);  %shift??
epidermis_mask2=imresize(epidermis_mask,[512 512]);

rootFolder4modification_mask512 = [hostfolder '\modification_mask512']; 
ImName_mask512 = [hostfolder '\modification_mask512\' num2str(fname3) '.png'] ;
rootFolder4modification_mask = [hostfolder '\modification_mask']; 
ImName_mask = [hostfolder '\modification_mask\' num2str(fname3) '.png'] ;
if exist(ImName_mask512)==0
    mkdir(rootFolder4modification_mask512);
end
if exist(ImName_mask)==0
    mkdir(rootFolder4modification_mask);
end
imwrite(epidermis_mask2,ImName_mask512,'png');
imwrite(epidermis_mask,ImName_mask,'png');

handles.output = hObject;
% Update handles structure
guidata(hObject, handles); 


% --------------------------------------------------------------------
function FileMenu_Callback(hObject, eventdata, handles)
% hObject    handle to FileMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function OpenMenuItem_Callback(hObject, eventdata, handles)
% hObject    handle to OpenMenuItem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
file = uigetfile('*.fig');
if ~isequal(file, 0)
    open(file);
end

% --------------------------------------------------------------------
function PrintMenuItem_Callback(hObject, eventdata, handles)
% hObject    handle to PrintMenuItem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
printdlg(handles.figure1)

% --------------------------------------------------------------------
function CloseMenuItem_Callback(hObject, eventdata, handles)
% hObject    handle to CloseMenuItem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
selection = questdlg(['Close ' get(handles.figure1,'Name') '?'],...
                     ['Close ' get(handles.figure1,'Name') '...'],...
                     'Yes','No','Yes');
if strcmp(selection,'No')
    return;
end

delete(handles.figure1)


% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1


% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
     set(hObject,'BackgroundColor','white');
end

set(hObject, 'String', {'plot(rand(5))', 'plot(sin(1:0.01:25))', 'bar(1:.5:10)', 'plot(membrane)', 'surf(peaks)'});


% --- Executes on button press in togglebutton1.
function togglebutton1_Callback(hObject, eventdata, handles) %revise_EDJ
% hObject    handle to togglebutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

cla;
global statu c 
Y_EDJ2=handles.Y_EDJ;
x_sel = [];
y_sel = [];
flagLoop = 1;
title('manual modification of EDJ','FontSize',16);
fig1=imagesc(handles.OAC),colormap(gray),hold on;
plot(handles.Y_EDJ,'b','LineWidth',1.2),hold on;
set(gca,'YDir','reverse'),axis([0 944 0 370]);
statu=1;
i=1;
str1='recover EDJ line';
str2='clean EDJ line';
text(20,350,str2,'Color','red','FontSize',30)
text(500,350,str1,'Color','green','FontSize',30)
while flagLoop || flagLoop1 || flagLoop2
    [x_sel_tmp y_sel_tmp] = ginput(1);
    flagLoop = ((x_sel_tmp >= 0) && (x_sel_tmp <= 944) && (y_sel_tmp >= 0) && (y_sel_tmp <= 340));
    flagLoop1 = ((x_sel_tmp >= 0) && (x_sel_tmp <= 472) && (y_sel_tmp >= 340) && (y_sel_tmp <= 370));
    flagLoop2 = ((x_sel_tmp >= 473) && (x_sel_tmp <= 944) && (y_sel_tmp >= 340) && (y_sel_tmp <= 370));
    if flagLoop1
        cla(fig1)
        text(20,350,str2,'Color','red','FontSize',30)
        text(500,350,str1,'Color','green','FontSize',30)
         if i>=2
             for j=1:i-1
                 xtemp=P1{1,j}(1);
                 ytemp=P1{1,j}(2);
                 plot(xtemp, ytemp , '.y', 'MarkerFaceColor', 'y'); hold on;
                 j=j+1;
             end
         end
     elseif flagLoop2
        plot(handles.Y_EDJ,'b','LineWidth',1.2),hold on;
     end
    if flagLoop
        plot(x_sel_tmp, y_sel_tmp, '.y', 'MarkerFaceColor', 'y'); hold on;
        P1(1,i)={[x_sel_tmp y_sel_tmp]};
        i=i+1;
        x_sel = [x_sel x_sel_tmp];
        y_sel = [y_sel y_sel_tmp];
    end

end

figure(602), plot(x_sel, y_sel),set(gca,'YDir','reverse');

x_1=round(x_sel);
y_1=round(y_sel);
xi=x_1(1):1:x_1(end);
%     n=11;
%     factor = 4;
%yi = interpft(y_1, factor*n);
%plot( xi, yi, '.-'),set(gca,'YDir','reverse');
yi = interp1(x_1, y_1, xi);
yi = round(yi);
%plot(xi,yi,'.-'),set(gca,'YDir','reverse');
   % revise surface
   n=1;
   for xiii=round(min(xi)):1:round(max(xi))
       Y_EDJ2(1,xiii) = yi(n);
       n=n+1;
   end
   handles.Y_EDJ2=Y_EDJ2;
   imagesc(handles.OAC),colormap(gray),hold on;
   
   plot(handles.Y_EDJ2,'b','LineWidth',1.2),hold on;
   title('manual modification of EDJ','FontSize',16);
   set(gca,'YDir','reverse'),axis([0 944 0 370]);


   handles.output = hObject;
   guidata(hObject, handles); 
% Hint: get(hObject,'Value') returns toggle state of togglebutton1


% --- Executes on button press in togglebutton2.
function togglebutton2_Callback(hObject, eventdata, handles) %revise_surface
% hObject    handle to togglebutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    cla;
    global Gy lines fname3 statu bon
    statu=2;
    Y_surface2=handles.Y_surface;
    x_sel = [];
    y_sel = [];
    flagLoop = 1;
    title('manual modification of surface','FontSize',16);
    fig=imagesc(handles.OAC);colormap(gray),hold on;
    pp = plot(handles.Y_surface,'r','LineWidth',1.2);hold on;
    set(gca,'YDir','reverse')
    axis([0 944 0 370]);
    i=1;
    str1='recover surface line';
    str2='clean surface line';
    text(20,350,str2,'Color','red','FontSize',30)
    text(500,350,str1,'Color','green','FontSize',30)
    while flagLoop || flagLoop1 || flagLoop2
        
        [x_sel_tmp y_sel_tmp] = ginput(1);
        
     flagLoop = ((x_sel_tmp >= 0) && (x_sel_tmp <= 944) && (y_sel_tmp >= 0) && (y_sel_tmp <= 340));
     flagLoop1 = ((x_sel_tmp >= 0) && (x_sel_tmp <= 472) && (y_sel_tmp >= 340) && (y_sel_tmp <= 370));
     flagLoop2 = ((x_sel_tmp >= 473) && (x_sel_tmp <= 944) && (y_sel_tmp >= 340) && (y_sel_tmp <= 370));
     if flagLoop1
         cla(fig)
         text(20,350,str2,'Color','red','FontSize',30)
         text(500,350,str1,'Color','green','FontSize',30)
         if i>=2
             for j=1:i-1
                 xtemp=P1{1,j}(1);
                 ytemp=P1{1,j}(2);
                 plot(xtemp, ytemp , '.y', 'MarkerFaceColor', 'y'); hold on;
                 j=j+1;
             end
         end
     elseif flagLoop2
        pp = plot(handles.Y_surface,'r','LineWidth',1.2);hold on;
     end
            
        if flagLoop
            plot(x_sel_tmp, y_sel_tmp, '.y', 'MarkerFaceColor', 'y'); hold on;
            P1(1,i)={[x_sel_tmp y_sel_tmp]};
            i=i+1;
            x_sel = [x_sel x_sel_tmp];
            y_sel = [y_sel y_sel_tmp];
        end

    end

     figure(10),plot(x_sel, y_sel),set(gca,'YDir','reverse');

    x_1s=round(x_sel);
    y_1s=round(y_sel);
    xis=x_1s(1):1:x_1s(end);
    %     n=11;
    %     factor = 4;
    %yi = interpft(y_1, factor*n);
    
    %plot( xi, yi, '.-'),set(gca,'YDir','reverse');
    yis = interp1(x_1s, y_1s, xis);
    yis = round(yis);
    %plot(xis,yis,'.-'),set(gca,'YDir','reverse');

   % revise surface
   n=1;
   for xii=round(min(xis)):1:round(max(xis))
       Y_surface2(1,xii) = yis(n);
       n=n+1;
   end
   handles.Y_surface2=Y_surface2;
   imagesc(handles.OAC),colormap(gray),hold on;
   plot(handles.Y_surface2,'r','LineWidth',1.2),hold on;
   title('manual modification of surface','FontSize',16);
   set(gca,'YDir','reverse'),axis([0 944 0 370]);
   

   Gy2=Gy;
   Y_EDJ2=handles.Y_EDJ;
      for xii=round(min(xis)):1:round(max(xis))
            q=Y_surface2(1,xii);
%             if  strfind(fname3,'hand3')>1
%             [M2,N2]=max(Gy2(q+15:q+80,xii));
%             Y_EDJ2(1,xii)=N2+q+15;
%             
%             elseif strfind(fname3,'hand4')>1
%             [M2,N2]=max(Gy2(q+15:q+80,xii));
%             Y_EDJ2(1,xii)=N2+q+15;
%             else
            [M2,N2]=max(Gy2(q+20:q+60,xii));
            Y_EDJ2(1,xii)=N2+q+20;

%             end
      end
   xtemp1=P1{1,1}(1);
   xtemp2=P1{1,i-1}(1);   
    xtemp1=round(xtemp1);
    xtemp2=round(xtemp2);
   if xtemp1 >= 6 && xtemp2 <= lines-10
       for x=xtemp1:1:xtemp2
           if abs(Y_EDJ2(:,x-1)-Y_EDJ2(:,x))>5
               Y_EDJ2(:,x)=(Y_EDJ2(:,x-3)+Y_EDJ2(:,x+3))/2;
           elseif abs(Y_EDJ2(:,x-3)-Y_EDJ2(:,x))>15
               Y_EDJ2(:,x)=(Y_EDJ2(:,x-5)+Y_EDJ2(:,x+5))/2; 
           else
               Y_EDJ2(:,x)=Y_EDJ2(:,x);
           end
       end
   else

       for x=6:1:lines-10
           if abs(Y_EDJ2(:,x-1)-Y_EDJ2(:,x))>5
               Y_EDJ2(:,x)=(Y_EDJ2(:,x-3)+Y_EDJ2(:,x+3))/2;
           elseif abs(Y_EDJ2(:,x-3)-Y_EDJ2(:,x))>15
               Y_EDJ2(:,x)=(Y_EDJ2(:,x-5)+Y_EDJ2(:,x+5))/2;
           else
               Y_EDJ2(:,x)=Y_EDJ2(:,x);
           end
       end
% 
   end
   handles.Y_EDJ2=Y_EDJ2;

   handles.output = hObject;
   guidata(hObject, handles); 
% Hint: get(hObject,'Value') returns toggle state of togglebutton2


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles) %next
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
cla;
statu=0;
if isfield(handles, 'Y_surface2')
    handles = rmfield(handles, 'Y_surface2');
end

if isfield(handles, 'Y_EDJ2')
    handles = rmfield(handles, 'Y_EDJ2');
end

global x lines binBase hostfolder fname3 Gy bon
handles.binBaseInd=handles.binBaseInd + 1;
binBaseInd=handles.binBaseInd;
if binBaseInd==length(binBase(:))+1
    msgbox('the end');
else

    fname = binBase(binBaseInd).name;
    underscore_inds = strfind(fname, '.');
    fname2=fname(1:underscore_inds-5);
    n1=str2num(fname(underscore_inds-4:underscore_inds-1));
    n1=n1/4;
    fname3=[fname2 num2str(n1,'%04d')];


    epidermis_struct=load([hostfolder '\epidermis_mat\' fname3 '.mat']);
    OAC_1024image=imread([hostfolder '\OAC_image_size1024\' fname]);
    handles.OAC=OAC_1024image;
    %
    [thr,sorh,keepapp] = ddencmp('den','wv',OAC_1024image);
    ixc=wdencmp('gbl',OAC_1024image,'sym4',5,thr,sorh,keepapp);
    k3=medfilt2(ixc,[15 15]);
    isuo=imresize(k3,1, 'bicubic' );
    [Gx, Gy] = imgradientxy(isuo,'sobel');
    X=zeros(1,lines);
    for x=1:1:lines
        [M,N]=max(Gy(1:200,x)); %set the boundary
        X(1,x)=N+bon;
    end

    for x=6:1:lines-10
        if abs(X(:,x-1)-X(:,x))>10
            X(:,x)=(X(:,x-3)+X(:,x+3))/2;
        elseif abs(X(:,x-3)-X(:,x))>20
            X(:,x)=(X(:,x-5)+X(:,x+5))/2;
        else
            X(:,x)=X(:,x);
        end
    end

    for x=1:1:50 %for bug of edges
        if abs(X(:,x)-X(:,x+10))>30
            X(:,x)=X(:,x+20);
        else
            X(:,x)=X(:,x);
        end
    end
%%
    [row,col,v]=find(epidermis_struct.epidermis_mat2);
    count_col=zeros(1,1000);
    y_2_surface=zeros(1,1000);
    y_2_EDJ=zeros(1,1000);
    for i=1:1:1000
        count_col(1,i)=sum(col(:)==i);

        if i>1
            y_2_surface(i)=row(sum(count_col(1,1:i-1))+1);
            y_2_EDJ(i)=row(sum(count_col(1,1:i)));
        else
            y_2_surface(i)=row(1);
            y_2_EDJ(i)=row(sum(count_col(1,1)));
        end

    end
    y_surface=y_2_surface(1,30:end-27);
    y_EDJ=y_2_EDJ(1,30:end-27);
    balance_y=round(mean(X-y_surface));
    Y_surface=y_surface+balance_y;
    Y_EDJ=y_EDJ+balance_y;
    handles.Y_surface=Y_surface;
    handles.Y_EDJ=Y_EDJ;
    newChr = extractAfter(fname3,strlength(fname3)-4);
    axes(handles.axes1);
    
    imagesc(handles.OAC),colormap(gray),hold on;
    plot(handles.Y_surface,'r','LineWidth',1.2),hold on;
    plot(handles.Y_EDJ,'b','LineWidth',1.2),hold on;
    title('manual modification #'+ string(newChr),'FontSize',22);
    set(gca,'YDir','reverse'),axis([0 944 0 370]);
    handles.output = hObject;
    guidata(hObject, handles);
end

% --- Executes on button press in pushbutton9.
function pushbutton8_Callback(hObject, eventdata, handles) %mask up
% hObject    handle to pushbutton9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
 cla;
statu=0;
if isfield(handles, 'Y_surface2')
    handles = rmfield(handles, 'Y_surface2');
end

if isfield(handles, 'Y_EDJ2')
    handles = rmfield(handles, 'Y_EDJ2');
end

global x lines binBase hostfolder fname3 Gy bon
bon = bon+1;

binBaseInd=handles.binBaseInd;


    fname = binBase(binBaseInd).name;
    underscore_inds = strfind(fname, '.');
    fname2=fname(1:underscore_inds-5);
    n1=str2num(fname(underscore_inds-4:underscore_inds-1));
    n1=n1/4;
    fname3=[fname2 num2str(n1,'%04d')];


    epidermis_struct=load([hostfolder '\epidermis_mat\' fname3 '.mat']);
    OAC_1024image=imread([hostfolder '\OAC_image_size1024\' fname]);
    handles.OAC=OAC_1024image;
    %
    [thr,sorh,keepapp] = ddencmp('den','wv',OAC_1024image);
    ixc=wdencmp('gbl',OAC_1024image,'sym4',5,thr,sorh,keepapp);
    k3=medfilt2(ixc,[15 15]);
    isuo=imresize(k3,1, 'bicubic' );
    [Gx, Gy] = imgradientxy(isuo,'sobel');
    X=zeros(1,lines);
    for x=1:1:lines
        [M,N]=max(Gy(1:200,x)); %set the boundary
        X(1,x)=N+bon;
    end

    for x=6:1:lines-10
        if abs(X(:,x-1)-X(:,x))>10
            X(:,x)=(X(:,x-3)+X(:,x+3))/2;
        elseif abs(X(:,x-3)-X(:,x))>20
            X(:,x)=(X(:,x-5)+X(:,x+5))/2;
        else
            X(:,x)=X(:,x);
        end
    end

    for x=1:1:50 %for bug of edges
        if abs(X(:,x)-X(:,x+10))>30
            X(:,x)=X(:,x+20);
        else
            X(:,x)=X(:,x);
        end
    end
%%
    [row,col,v]=find(epidermis_struct.epidermis_mat2);
    count_col=zeros(1,1000);
    y_2_surface=zeros(1,1000);
    y_2_EDJ=zeros(1,1000);
    for i=1:1:1000
        count_col(1,i)=sum(col(:)==i);

        if i>1
            y_2_surface(i)=row(sum(count_col(1,1:i-1))+1);
            y_2_EDJ(i)=row(sum(count_col(1,1:i)));
        else
            y_2_surface(i)=row(1);
            y_2_EDJ(i)=row(sum(count_col(1,1)));
        end

    end
    y_surface=y_2_surface(1,30:end-27);
    y_EDJ=y_2_EDJ(1,30:end-27);
    balance_y=round(mean(X-y_surface));
    Y_surface=y_surface+balance_y;
    Y_EDJ=y_EDJ+balance_y;
    handles.Y_surface=Y_surface;
    handles.Y_EDJ=Y_EDJ;
    newChr = extractAfter(fname3,strlength(fname3)-4);
    axes(handles.axes1);
    
    imagesc(handles.OAC),colormap(gray),hold on;
    plot(handles.Y_surface,'r','LineWidth',1.2),hold on;
    plot(handles.Y_EDJ,'b','LineWidth',1.2),hold on;
    title('manual modification #'+ string(newChr),'FontSize',22);
    set(gca,'YDir','reverse'),axis([0 944 0 370]);
    handles.output = hObject;
    guidata(hObject, handles);

%modification_gui_OpeningFcn();
% --- Executes on button press in pushbutton9.
function pushbutton9_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
 cla;
statu=0;
if isfield(handles, 'Y_surface2')
    handles = rmfield(handles, 'Y_surface2');
end

if isfield(handles, 'Y_EDJ2')
    handles = rmfield(handles, 'Y_EDJ2');
end

global x lines binBase hostfolder fname3 Gy bon
bon = bon-1;

binBaseInd=handles.binBaseInd;


    fname = binBase(binBaseInd).name;
    underscore_inds = strfind(fname, '.');
    fname2=fname(1:underscore_inds-5);
    n1=str2num(fname(underscore_inds-4:underscore_inds-1));
    n1=n1/4;
    fname3=[fname2 num2str(n1,'%04d')];


    epidermis_struct=load([hostfolder '\epidermis_mat\' fname3 '.mat']);
    OAC_1024image=imread([hostfolder '\OAC_image_size1024\' fname]);
    handles.OAC=OAC_1024image;
    %
    [thr,sorh,keepapp] = ddencmp('den','wv',OAC_1024image);
    ixc=wdencmp('gbl',OAC_1024image,'sym4',5,thr,sorh,keepapp);
    k3=medfilt2(ixc,[15 15]);
    isuo=imresize(k3,1, 'bicubic' );
    [Gx, Gy] = imgradientxy(isuo,'sobel');
    X=zeros(1,lines);
    for x=1:1:lines
        [M,N]=max(Gy(1:200,x)); %set the boundary
        X(1,x)=N+bon;
    end

    for x=6:1:lines-10
        if abs(X(:,x-1)-X(:,x))>10
            X(:,x)=(X(:,x-3)+X(:,x+3))/2;
        elseif abs(X(:,x-3)-X(:,x))>20
            X(:,x)=(X(:,x-5)+X(:,x+5))/2;
        else
            X(:,x)=X(:,x);
        end
    end

    for x=1:1:50 %for bug of edges
        if abs(X(:,x)-X(:,x+10))>30
            X(:,x)=X(:,x+20);
        else
            X(:,x)=X(:,x);
        end
    end
%%
    [row,col,v]=find(epidermis_struct.epidermis_mat2);
    count_col=zeros(1,1000);
    y_2_surface=zeros(1,1000);
    y_2_EDJ=zeros(1,1000);
    for i=1:1:1000
        count_col(1,i)=sum(col(:)==i);

        if i>1
            y_2_surface(i)=row(sum(count_col(1,1:i-1))+1);
            y_2_EDJ(i)=row(sum(count_col(1,1:i)));
        else
            y_2_surface(i)=row(1);
            y_2_EDJ(i)=row(sum(count_col(1,1)));
        end

    end
    y_surface=y_2_surface(1,30:end-27);
    y_EDJ=y_2_EDJ(1,30:end-27);
    balance_y=round(mean(X-y_surface));
    Y_surface=y_surface+balance_y;
    Y_EDJ=y_EDJ+balance_y;
    handles.Y_surface=Y_surface;
    handles.Y_EDJ=Y_EDJ;
    newChr = extractAfter(fname3,strlength(fname3)-4);
    axes(handles.axes1);
    
    imagesc(handles.OAC),colormap(gray),hold on;
    plot(handles.Y_surface,'r','LineWidth',1.2),hold on;
    plot(handles.Y_EDJ,'b','LineWidth',1.2),hold on;
    title('manual modification #'+ string(newChr),'FontSize',22);
    set(gca,'YDir','reverse'),axis([0 944 0 370]);
    handles.output = hObject;
    guidata(hObject, handles);

%modification_gui_OpeningFcn();
