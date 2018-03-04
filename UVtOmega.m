function y = UVtOmega(U,V,I,J,col); 

y = zeros(length(I), 1);
[m, n] = size(V);
if length(col) <= 1
    y = -1;
else
    for k = 1:length(col)-1
        j = J(col(k)+1);
        if j > m
            y = -1;
            break;
        end
        Xj = U * V(j,:)';
        idx = [col(k)+1:col(k+1)];
        if max(col(k)+1, col(k+1)) > length(I)
            y = -1;
            break;
        end
        y(idx) = Xj(I(idx));
    end
end
% for j = 1:length(col)-1
%     Xj = U * V(j,:)';
%     idx = [col(j)+1:col(j+1)];
%     y(idx) = Xj(I(idx));
% end


