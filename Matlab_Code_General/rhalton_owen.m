function U = rhalton_owen(num_samp, d, seed, doShift)
% U: d x num_samp in [0,1); Owen-scrambled Halton (+ optional CP shift)
if nargin < 3, seed = 12345; end
if nargin < 4, doShift = true; end

pr = firstprimes(d);
U  = zeros(d, num_samp);

for j = 1:d
    b = pr(j);
    for n = 1:num_samp
        x = 0; invb = 1; m = n; prefix = 0; L = 0;
        while m > 0
            L    = L + 1;
            invb = invb / b;
            digit = mod(m, b); m = floor(m / b);
            % Owen nested scrambling: permutation depends on (j,L,prefix)
            sd = perm_digit(b, seed, j, L, prefix, digit);
            x  = x + double(sd) * invb;
            prefix = prefix * b + double(sd); % update prefix for next level
        end
        U(j, n) = x;
    end
end

% Optional Cranley–Patterson shift (independent of digit permutations)
if doShift
    s = RandStream('mt19937ar','Seed', seed + 99991);
    shift = rand(s, d, 1);
    U = U + shift; U = U - floor(U);
end
end

function sd = perm_digit(b, seed, j, L, prefix, digit)
% Return scrambled digit in {0,...,b-1} via a permutation keyed by (j,L,prefix)
s = RandStream('mt19937ar','Seed', hash32(seed, j, L, prefix));
perm = randperm(s, b) - 1;          % row vector of 0..b-1
sd = perm(digit + 1);
end

function h = hash32(a,b,c,d)
% Simple FNV-like 32-bit mixer → MATLAB seedable integer
h = uint32(2166136261);
h = uint32(16777619) * bitxor(h, uint32(a));
h = uint32(16777619) * bitxor(h, uint32(b));
h = uint32(16777619) * bitxor(h, uint32(c));
h = uint32(16777619) * bitxor(h, uint32(d));
h = double(bitand(h, uint32(2^31-1))); % keep within RandStream seed range
end

function p = firstprimes(k)
p = zeros(1,k); n = 2; i = 1;
while i <= k
    if isprime(n), p(i) = n; i = i + 1; end
    n = n + 1;
end
end
