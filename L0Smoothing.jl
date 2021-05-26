using Images, TestImages, ImageFiltering, MosaicViews, BenchmarkTools
using FFTW

function my_diffy(img)
    d = similar(img)
    d[:, 1:end-1, :] .= @views img[:, 2:end, :] .- img[:, 1:end-1, :]
    d[:, end:end, :] .= @views img[:, 1:1, :] .- img[:, end:end, :]
    d
end

function my_diffyt(img)
    d = similar(img)
    d[:, 2:end, :] .= @views img[:, 1:end-1, :] .- img[:, 2:end, :]
    d[:, 1:1, :] .= @views img[:, end:end, :] .- img[:, 1:1, :]
    d
end

function my_diffx(img)
    d = similar(img)
    d[:, :, 1:end-1] .= @views img[:, :, 2:end] .- img[:, :, 1:end-1]
    d[:, :, end:end] .= @views img[:, :, 1:1] .- img[:, :, end:end]
    d
end

function my_diffxt(img)
    d = similar(img)
    d[:, :, 2:end] .= @views img[:, :, 1:end-1] .- img[:, :, 2:end]
    d[:, :, 1:1] .= @views img[:, :, end:end] .- img[:, :, 1:1]
    d
end

expanded_channelview(img::AbstractArray{T}) where T<:Colorant = channelview(img)
expanded_channelview(img::AbstractArray{T}) where T<:Gray = reshape(channelview(img), 1, size(img)...)

function l0smoothing(img,  λ=2e-2, κ=2.0)
    S = float.(expanded_channelview(img))
    βmax = 1e5
    fx = [1 -1]
    fy = [1, -1]
    D, N, M = size(S)
    sizeI2D = (N, M)
    sizeI2D_t = (M, N)
    otfFx = freqkernel(centered(fx), sizeI2D)
    otfFy = transpose(freqkernel(centered(transpose(fy)), sizeI2D_t))
    Normin1 = fft(S, (2, 3))
    Denormin2 = abs.(otfFx).^2 + abs.(otfFy ).^2
    if D > 1
        Denormin2 = repeat(reshape(Denormin2, 1, size(Denormin2)...), inner=(1, 1, 1), outer=(3, 1, 1))
    else
        Denormin2 = reshape(Denormin2, 1, size(Denormin2)...)
    end
    β = 2*λ
    while β < βmax
        Denormin = 1 .+ β*Denormin2

        h = my_diffx(S)
        v = my_diffy(S)

        if D == 1
            t = (h.^2+v.^2) .< λ/β
        else
            t = sum((h.^2+v.^2), dims=1) .< λ/β
            t = repeat(t, inner=(1, 1, 1), outer=(D, 1, 1))
        end

        h[t] .= 0
        v[t] .= 0

        Normin2 = my_diffxt(h)
        Normin2 = Normin2 + my_diffyt(v)
        FS = (Normin1 + β*fft(Normin2, (2, 3))) ./ Denormin
        S = real(ifft(FS, (2, 3)))
        β = β*κ
    end
    if D == 1
        return colorview(Gray, S[1,:,:])
    else
        return colorview(RGB, S)
    end
end

img1 = load("pflower.jpg")
img2 = testimage("cameraman")

S1 = @btime l0smoothing(img1)
S2 = @btime l0smoothing(img2)

mosaicview(img1,img2, S1, S2;nrow=2, ncol=2, rowmajor=true)
