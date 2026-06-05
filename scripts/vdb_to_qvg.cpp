// vdb_to_qvg.cpp — convert an OpenVDB .vdb file to the Quasi .qvg
// density-grid format the path tracer reads.
//
// Equivalent to scripts/vdb_to_qvg.py but uses OpenVDB's C++ API
// directly so we don't depend on pyopenvdb (which isn't packaged
// on PyPI and is a real pain to build from source). Compile with:
//
//   c++ -std=c++17 -O2 \
//       -I$(brew --prefix openvdb)/include \
//       -I$(brew --prefix tbb)/include \
//       -L$(brew --prefix openvdb)/lib \
//       -lopenvdb \
//       scripts/vdb_to_qvg.cpp -o scripts/vdb_to_qvg
//
// Usage:
//
//   ./scripts/vdb_to_qvg INPUT.vdb OUTPUT.qvg \
//       [--resolution N]              # N x N x N (default 64)
//       [--grid-name NAME]            # default "density"
//       [--normalize]                 # rescale max -> 1.0
//       [--list-grids]                # print grid names + exit
//
// The .qvg format matches src/pathtrace/grid.rs::Grid3D::load
// byte-for-byte (little-endian throughout).

#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <openvdb/tools/Interpolation.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string input;
    std::string output;
    int res_x = 64;
    int res_y = 64;
    int res_z = 64;
    std::string grid_name = "density";
    bool normalize = false;
    bool list_grids = false;
};

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " INPUT.vdb OUTPUT.qvg [options]\n"
              << "Options:\n"
              << "  --resolution N         output dims N x N x N (default 64)\n"
              << "  --resolution X Y Z     non-uniform output dims\n"
              << "  --grid-name NAME       scalar grid to read (default \"density\")\n"
              << "  --normalize            rescale max -> 1.0\n"
              << "  --list-grids           print grid names and exit\n";
}

bool parse_args(int argc, char** argv, Args& a) {
    if (argc < 2) {
        return false;
    }
    a.input = argv[1];
    int i = 2;
    if (i < argc && argv[i][0] != '-') {
        a.output = argv[i++];
    }
    while (i < argc) {
        std::string s = argv[i++];
        if (s == "--resolution") {
            if (i >= argc) return false;
            int n0 = std::stoi(argv[i++]);
            if (i + 1 < argc && argv[i][0] != '-' && argv[i + 1][0] != '-') {
                // Non-uniform: three values
                int n1 = std::stoi(argv[i++]);
                int n2 = std::stoi(argv[i++]);
                a.res_x = n0; a.res_y = n1; a.res_z = n2;
            } else {
                a.res_x = a.res_y = a.res_z = n0;
            }
        } else if (s == "--grid-name") {
            if (i >= argc) return false;
            a.grid_name = argv[i++];
        } else if (s == "--normalize") {
            a.normalize = true;
        } else if (s == "--list-grids") {
            a.list_grids = true;
        } else if (s == "--help" || s == "-h") {
            return false;
        } else {
            std::cerr << "unknown option: " << s << "\n";
            return false;
        }
    }
    if (!a.list_grids && a.output.empty()) {
        std::cerr << "output path required (unless --list-grids)\n";
        return false;
    }
    return true;
}

void write_le_u32(std::ostream& os, uint32_t v) {
    char b[4];
    b[0] = static_cast<char>(v & 0xff);
    b[1] = static_cast<char>((v >> 8) & 0xff);
    b[2] = static_cast<char>((v >> 16) & 0xff);
    b[3] = static_cast<char>((v >> 24) & 0xff);
    os.write(b, 4);
}

void write_le_f32(std::ostream& os, float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    write_le_u32(os, bits);
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 2;
    }
    openvdb::initialize();

    openvdb::io::File file(args.input);
    try {
        file.open();
    } catch (const openvdb::IoError& e) {
        std::cerr << "failed to open " << args.input << ": " << e.what() << "\n";
        return 1;
    }

    if (args.list_grids) {
        std::cout << args.input << ": " << file.getGrids()->size() << " grid(s)\n";
        for (auto it = file.beginName(); it != file.endName(); ++it) {
            std::cout << "  " << *it << "\n";
        }
        file.close();
        return 0;
    }

    openvdb::GridBase::Ptr base_grid;
    try {
        base_grid = file.readGrid(args.grid_name);
    } catch (const openvdb::Exception& e) {
        std::cerr << "failed to read grid '" << args.grid_name << "': " << e.what() << "\n";
        std::cerr << "available grids:\n";
        for (auto it = file.beginName(); it != file.endName(); ++it) {
            std::cerr << "  " << *it << "\n";
        }
        return 1;
    }
    file.close();

    auto float_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(base_grid);
    if (!float_grid) {
        std::cerr << "grid '" << args.grid_name << "' is not a scalar FloatGrid\n";
        return 1;
    }

    // World-space bounding box of the active voxels.
    openvdb::CoordBBox bbox = float_grid->evalActiveVoxelBoundingBox();
    auto transform = float_grid->transformPtr();
    openvdb::Vec3d ws_min = transform->indexToWorld(bbox.min().asVec3d());
    openvdb::Vec3d ws_max = transform->indexToWorld(bbox.max().asVec3d());

    // Sample at the centre of each output voxel using trilinear
    // world-space interpolation. The procedural script does the same
    // thing via pyopenvdb's GridSampler.wsSample.
    auto accessor = float_grid->getConstAccessor();
    openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler>
        sampler(accessor, *transform);

    const int W = args.res_x;
    const int H = args.res_y;
    const int D = args.res_z;
    const size_t voxel_count = static_cast<size_t>(W) * H * D;
    std::vector<float> values(voxel_count, 0.0f);
    float max_val = 0.0f;
    size_t idx = 0;
    for (int iz = 0; iz < D; ++iz) {
        double z = ws_min.z() + (iz + 0.5) * (ws_max.z() - ws_min.z()) / D;
        for (int iy = 0; iy < H; ++iy) {
            double y = ws_min.y() + (iy + 0.5) * (ws_max.y() - ws_min.y()) / H;
            for (int ix = 0; ix < W; ++ix) {
                double x = ws_min.x() + (ix + 0.5) * (ws_max.x() - ws_min.x()) / W;
                float v = sampler.wsSample(openvdb::Vec3d(x, y, z));
                if (v < 0.0f) v = 0.0f;
                if (v > max_val) max_val = v;
                values[idx++] = v;
            }
        }
    }

    float scale = 1.0f;
    if (args.normalize && max_val > 0.0f) {
        scale = 1.0f / max_val;
    }

    std::vector<uint8_t> voxels(voxel_count, 0);
    size_t nonzero = 0;
    double sum = 0.0;
    for (size_t i = 0; i < voxel_count; ++i) {
        float v = values[i] * scale;
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        uint8_t q = static_cast<uint8_t>(v * 255.0f + 0.5f);
        voxels[i] = q;
        if (q > 0) ++nonzero;
        sum += static_cast<double>(q);
    }
    double mean_density = sum / voxel_count / 255.0;

    std::ofstream out(args.output, std::ios::binary);
    if (!out) {
        std::cerr << "failed to open " << args.output << " for write\n";
        return 1;
    }
    out.write("QVG1", 4);
    write_le_u32(out, static_cast<uint32_t>(W));
    write_le_u32(out, static_cast<uint32_t>(H));
    write_le_u32(out, static_cast<uint32_t>(D));
    write_le_f32(out, static_cast<float>(ws_min.x()));
    write_le_f32(out, static_cast<float>(ws_min.y()));
    write_le_f32(out, static_cast<float>(ws_min.z()));
    write_le_f32(out, static_cast<float>(ws_max.x()));
    write_le_f32(out, static_cast<float>(ws_max.y()));
    write_le_f32(out, static_cast<float>(ws_max.z()));
    write_le_u32(out, static_cast<uint32_t>(voxel_count));
    out.write(reinterpret_cast<const char*>(voxels.data()), voxels.size());
    if (!out) {
        std::cerr << "failed to write " << args.output << "\n";
        return 1;
    }

    std::printf(
        "wrote %s (%zu bytes, dims=(%d,%d,%d), bbox_min=(%.3f,%.3f,%.3f), "
        "bbox_max=(%.3f,%.3f,%.3f), mean density=%.3f, non-zero voxels=%zu/%zu, "
        "max value=%.3f%s)\n",
        args.output.c_str(),
        static_cast<size_t>(44 + voxel_count),
        W, H, D,
        ws_min.x(), ws_min.y(), ws_min.z(),
        ws_max.x(), ws_max.y(), ws_max.z(),
        mean_density, nonzero, voxel_count,
        max_val,
        args.normalize ? " (normalized)" : ""
    );
    return 0;
}
