table = [
    "<15", "high", "no", "bad", "no";
    "<15", "high", "no", "good", "no";
    "16-25", "high", "no", "bad", "yes"
    ">26", "medium", "no", "bad", "yes"
    ">26", "low", "yes", "bad", "yes"
    ">26", "low", "yes", "good", "no"
    "16-25", "low", "yes", "good", "yes"
    "<15", "medium", "no", "bad", "no"
    "<15", "low", "yes", "bad", "yes"
    ">26", "medium", "yes", "bad", "yes"
    "<15", "medium", "yes", "good", "yes"
    "16-25", "medium", "no", "good", "yes"
    "16-25", "high", "yes", "bad", "yes"
    "<15", "medium", "no", "good", "no"];
%最一開始overall的entropy
entropy_bbq = -((9/14)*log2(9/14)+(5/14)*log2(5/14));

%第一層node，有4種可能，分別算出來
[attri_temp,entropy_temp,table_temp] = node(table,1,5);
gain_temp = gain(entropy_bbq, entropy_temp,attri_temp,table);

[attri_mid,entropy_mid,table_mid] = node(table,2,5);
gain_mid = gain(entropy_bbq, entropy_mid,attri_mid,table);

[attri_hw,entropy_hw,table_hw] = node(table,3,5);
gain_hw = gain(entropy_bbq, entropy_hw,attri_hw,table);

[attri_qz,entropy_qz,table_qz] = node(table,4,5);
gain_qz = gain(entropy_bbq, entropy_qz,attri_qz,table);

%gain_temp = 0.3149, gain_mid = 0.0292, gain_hw = 0.1518, gain_qz = 0.0481
% gain_temp是4個之中最大的，所以第一個node用temp往下分

%%
%第二層node1，由於15-26的部分已完全分開，所以只有<15與>26。

% temp<15 往下的分支
[attri_temp15_mid,entropy_temp15_mid,table_temp15_mid] = node(table_temp(:,:,2),2,5);
gain_temp15_mid = gain(entropy_temp(2), entropy_temp15_mid, attri_temp15_mid,table_temp(:,:,2)+table_temp(:,:,3));

[attri_temp15_hw,entropy_temp15_hw,table_temp15_hw] = node(table_temp(:,:,2),3,5);
gain_temp15_hw = gain(entropy_temp(2), entropy_temp15_hw, attri_temp15_hw,table_temp(:,:,2)+table_temp(:,:,3));

[attri_temp15_qz,entropy_temp15_qz,table_temp15_qz] = node(table_temp(:,:,2),4,5);
gain_temp15_qz = gain(entropy_temp(2), entropy_temp15_qz, attri_temp15_qz,table_temp(:,:,2)+table_temp(:,:,3));

%gain_temp15_mid = 0.6428, gain_temp15_hw = 0.9183, gain_temp15_qz = 0.3673
%在temp<15的分支下，hw的gain最大，在確認table_temp15_hw後發現已完全分開，到此結束

% temp>26 往下的分支
[attri_temp26_mid,entropy_temp26_mid,table_temp26_mid] = node(table_temp(:,:,3),2,5);
gain_temp26_mid = gain(entropy_temp(3), entropy_temp26_mid, attri_temp26_mid,table_temp(:,:,2)+table_temp(:,:,3));

[attri_temp26_hw,entropy_temp26_hw,table_temp26_hw] = node(table_temp(:,:,3),3,5);
gain_temp26_hw = gain(entropy_temp(3), entropy_temp26_hw, attri_temp26_hw,table_temp(:,:,2)+table_temp(:,:,3));

[attri_temp26_qz,entropy_temp26_qz,table_temp26_qz] = node(table_temp(:,:,3),4,5);
gain_temp26_qz = gain(entropy_temp(3), entropy_temp26_qz, attri_temp26_qz,table_temp(:,:,2)+table_temp(:,:,3));

%gain_temp26_mid = 0.0.6113, gain_temp26_hw = 0.5358, gain_temp26_qz =0.8113
%在temp>26的分支下，qz的gain最大，確認table_temp26_qz發現已完全分開，到此結束

%結論:
%if (15 < temp < 26)          then Yes
%if (temp < 15)&&( hw = yes)  then Yes
%if (temp < 15)&&( hw = no)   then No
%if (temp > 26)&&( qz = bad)  then Yes
%if (temp > 26)&&( qz = good) then No

%%
%node function是用來算一個node下其每個可能分支的entropy
function [attri, entropy, newtable] = node(table, colume, y)
data = table(:,colume);
tag = table(:,y);
types = unique(data);
attri = strings(length(data),2,length(types));
newtable = strings(size(table,1),size(table,2),length(types));
for i=1:length(data)
    for j=1:length(types)
        if data(i)==types(j)
            attri(i,1,j) = data(i);
            attri(i,2,j) = tag(i);
            newtable(i,:,j) = table(i,:);
        end
    end
end
entropy = zeros(1,length(types));
count_yes = 0;
count_no = 0;
for i=1:length(types)
    for j=1:length(data)
        if attri(j,2,i) == "yes"
            count_yes = count_yes + 1;
        elseif attri(j,2,i) == "no"
            count_no = count_no + 1;
        end
    end
    entropy(i)=-((count_yes/(count_yes+count_no))*log2(count_yes/(count_yes+count_no)) + (count_no/(count_yes+count_no))*log2(count_no/(count_yes+count_no)));
    if isnan(entropy(i))
        entropy(i) = 0;
    end
    count_yes = 0;
    count_no = 0;
end
end

%就是算gain的function
function gain_x = gain(entropy, entropy_x, attri, root)
ratio = zeros(1,size(entropy_x,2));
total = 0;
for i=1:size(root,1)
    if length(size(root)) == 3
        if root(i,2,1) ~= ""
            total = total+1;
        end
    else
        if root(i,2) ~= ""
            total = total+1;
        end
    end
end
for i=1:size(attri,3)
    for j=1:size(attri,1)
        if attri(j,2,i) ~= ""
            ratio(i) = ratio(i) + 1;
        end
    end
end
ratio = ratio/total;
gain_x = entropy;
for i=1:length(ratio)
    gain_x = gain_x - entropy_x(i)*ratio(i);
end
end