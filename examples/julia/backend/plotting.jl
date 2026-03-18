using CairoMakie
using GeometryBasics
using Meshing

const NLOLIB_HDR_POSITIONS = [0.00, 0.03, 0.30, 0.70, 0.95, 1.00]
const NLOLIB_HDR_COLORS = ["#ffffff", "#7ee3ed", "#4e8ec3", "#4d2d99", "#fd5ddd", "#ff207d"]
const NLOLIB_LINE_COLORS = ["#00b600", "#004b96", "#4391d1", "#ff2020", "#4c20b4"]
const NLOLIB_FIGURE_SIZE = (1200, 792)
const NLOLIB_TENSOR_FIGURE_SIZE = (1320, 960)

function nlolib_hdr_colormap()
    return cgrad(NLOLIB_HDR_COLORS, NLOLIB_HDR_POSITIONS)
end

function example_serif_font()
    if Sys.iswindows()
        return "Times New Roman"
    end
    return "DejaVu Serif"
end

function activate_example_theme!()
    set_theme!(
        Theme(
            palette = (; color = NLOLIB_LINE_COLORS),
            font = example_serif_font(),
            fontsize = 28,
            figure_padding = 18,
            backgroundcolor = :white,
            Axis = (
                xgridvisible = true,
                ygridvisible = true,
                xgridcolor = (:black, 0.18),
                ygridcolor = (:black, 0.18),
                topspinevisible = false,
                rightspinevisible = false,
            ),
            Axis3 = (
                xgridvisible = true,
                ygridvisible = true,
                zgridvisible = true,
                xgridcolor = (:black, 0.12),
                ygridcolor = (:black, 0.12),
                zgridcolor = (:black, 0.12),
                protrusions = (44, 26, 18, 28),
            ),
            Legend = (
                framevisible = false,
                backgroundcolor = :transparent,
            ),
            Colorbar = (
                ticklabelsize = 24,
                labelsize = 26,
            ),
            Lines = (
                linewidth = 3.0,
            ),
            Heatmap = (
                colormap = nlolib_hdr_colormap(),
            ),
        ),
    )
    return nothing
end

function styled_figure(; tensor::Bool = false)
    return Figure(size = tensor ? NLOLIB_TENSOR_FIGURE_SIZE : NLOLIB_FIGURE_SIZE)
end

function save_example_figure(path::AbstractString, fig)
    mkpath(dirname(path))
    save(path, fig)
    return path
end

function _normalized_nonnegative_data(values; normalization_peak = nothing)
    data = clamp.(Float64.(values), 0.0, Inf)
    data = replace(data, NaN => 0.0, Inf => 0.0, -Inf => 0.0)
    peak = maximum(data)
    norm_peak = normalization_peak === nothing ? peak : Float64(normalization_peak)
    if normalization_peak !== nothing && norm_peak <= 0.0
        error("normalization_peak must be positive when provided.")
    end
    if norm_peak > 0.0
        data = clamp.(data ./ norm_peak, 0.0, 1.0)
    end
    return data, norm_peak
end

function normalized_plot_data(values; normalization_peak = nothing)
    data, _ = _normalized_nonnegative_data(values; normalization_peak = normalization_peak)
    return data
end

function _evenly_spaced_indices(count::Integer; max_count::Integer)
    count > 0 || return Int[]
    max_count > 0 || error("max_count must be positive.")
    if count <= max_count
        return collect(1:count)
    end
    values = round.(Int, range(1, count, length = max_count))
    return unique(values)
end

function plot_3d_intensity_contours_propagation(
    x_axis,
    y_axis,
    z_axis,
    field_records,
    output_path;
    intensity_cutoff::Real = 0.05,
    num_levels::Integer = 7,
    max_x_samples::Integer = 48,
    max_y_samples::Integer = 48,
    max_z_samples::Integer = 18,
    alpha_min::Real = 0.12,
    alpha_max::Real = 0.72,
    input_is_intensity::Bool = false,
    normalization_peak = nothing,
    z_label::AbstractString = "z",
)
    0.0 <= intensity_cutoff < 1.0 || error("intensity_cutoff must be in [0, 1).")
    num_levels > 0 || error("num_levels must be positive.")
    max_x_samples > 0 || error("max_x_samples must be positive.")
    max_y_samples > 0 || error("max_y_samples must be positive.")
    max_z_samples > 0 || error("max_z_samples must be positive.")
    0.0 <= alpha_min <= alpha_max <= 1.0 || error("alpha_min/alpha_max must satisfy 0 <= alpha_min <= alpha_max <= 1.")

    x = Float64.(x_axis)
    y = Float64.(y_axis)
    z = Float64.(z_axis)
    raw = input_is_intensity ? Float64.(field_records) : abs2.(ComplexF64.(field_records))
    intensity, _ = _normalized_nonnegative_data(raw; normalization_peak = normalization_peak)

    ndims(intensity) == 3 || error("field_records must be [record, y, x].")
    size(intensity, 1) == length(z) || error("z axis length must match field_records size(record).")
    size(intensity, 2) == length(y) || error("y axis length must match field_records size(y).")
    size(intensity, 3) == length(x) || error("x axis length must match field_records size(x).")

    if maximum(intensity) <= 0.0
        println("intensity is zero everywhere; skipping 3D propagation contour-surface plot.")
        return nothing
    end

    x_indices = _evenly_spaced_indices(length(x); max_count = max_x_samples)
    y_indices = _evenly_spaced_indices(length(y); max_count = max_y_samples)
    z_indices = _evenly_spaced_indices(length(z); max_count = max_z_samples)
    x_small = x[x_indices]
    y_small = y[y_indices]
    z_small = z[z_indices]
    volume_data = permutedims(intensity[z_indices, y_indices, x_indices], (3, 2, 1))
    max_intensity = maximum(volume_data)
    if max_intensity < Float64(intensity_cutoff)
        println("no contours passed intensity cutoff; skipping 3D propagation contour-surface plot.")
        return nothing
    end
    cmap = nlolib_hdr_colormap()
    cmap_samples = CairoMakie.to_colormap(cmap)
    x_range = range(minimum(x_small), maximum(x_small), length = length(x_small))
    y_range = range(minimum(y_small), maximum(y_small), length = length(y_small))
    z_range = range(minimum(z_small), maximum(z_small), length = length(z_small))
    level_upper = min(0.92, max_intensity)
    if level_upper <= Float64(intensity_cutoff)
        println("no contours passed intensity cutoff; skipping 3D propagation contour-surface plot.")
        return nothing
    end
    levels = collect(range(Float64(intensity_cutoff), level_upper, length = num_levels))

    fig = styled_figure(tensor = true)
    x_span = max(maximum(x_small) - minimum(x_small), 1.0e-9)
    y_span = max(maximum(y_small) - minimum(y_small), 1.0e-9)
    z_span = max(maximum(z_small) - minimum(z_small), 1.0e-9)
    xy_scale = max(x_span, y_span, 1.0e-9)
    z_display_span = max(0.65 * xy_scale, min(z_span, 2.8 * xy_scale))
    ax = Axis3(
        fig[1, 1],
        xlabel = "x",
        ylabel = "y",
        zlabel = z_label,
        azimuth = -0.95,
        elevation = 0.55,
        perspectiveness = 0.0,
        aspect = (x_span / xy_scale, y_span / xy_scale, z_display_span / xy_scale),
    )
    any_surface = false
    for level in levels
        level <= max_intensity || continue
        vertices, faces = isosurface(volume_data, MarchingCubes(iso = level), x_range, y_range, z_range)
        isempty(vertices) && continue
        color_idx = clamp(round(Int, level * (length(cmap_samples) - 1)) + 1, 1, length(cmap_samples))
        color = cmap_samples[color_idx]
        mesh_obj = GeometryBasics.Mesh(
            Point3f[(Float32(v[1]), Float32(v[2]), Float32(v[3])) for v in vertices],
            TriangleFace{Int32}[TriangleFace{Int32}(Int32(f[1]), Int32(f[2]), Int32(f[3])) for f in faces],
        )
        mesh!(
            ax,
            mesh_obj;
            color = RGBAf(color.r, color.g, color.b, Float32(alpha_min + (alpha_max - alpha_min) * level)),
            shading = false,
            transparency = true,
            fxaa = false,
            overdraw = false,
        )
        any_surface = true
    end
    if !any_surface
        println("no contours passed intensity cutoff; skipping 3D propagation contour-surface plot.")
        return nothing
    end

    Colorbar(fig[1, 2], limits = (0.0, 1.0), colormap = cmap, label = "Normalized intensity")
    xlims!(ax, minimum(x_small), maximum(x_small))
    ylims!(ax, minimum(y_small), maximum(y_small))
    zlims!(ax, minimum(z_small), maximum(z_small))
    return save_example_figure(output_path, fig)
end


function plot_3d_intensity_volume_propagation(args...; kwargs...)
    return plot_3d_intensity_contours_propagation(args...; kwargs...)
end
