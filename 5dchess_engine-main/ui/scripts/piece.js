import { reportLoaded } from './startup.js';
const LOD_SIZES = [32, 16, 8, 4, 2, 1];

// Load a single SVG as Image
function loadSvgImage(filename) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.src = filename;
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error(`Failed to load: ${filename}`));
    });
}

// Cache a separate raster canvas for each piece and size. Once drawn, these
// canvases remain unchanged and can be used directly by drawImage at any zoom.
function rasterizeImage(img, size) {
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    canvas.getContext('2d').drawImage(img, 0, 0, size, size);
    return canvas;
}

async function loadAllSvgs() {
    const svgImages = {
        img: {}, // original SVGs
        lod: {}  // { size: { name: canvas, ... } }
    };

    // Initialize lod object for each size
    for (const size of LOD_SIZES) {
        svgImages.lod[size] = {};
    }

    // Load all SVGs
    const tasks = Object.entries(window.svg_file_mapping).map(
        async ([name, filename]) => {
            const img = await loadSvgImage(filename);
            svgImages.img[name] = img;

            // Downsample the previous raster so tiny levels retain coverage
            // instead of sampling a potentially transparent point in the SVG.
            let rasterSource = img;
            for (const size of LOD_SIZES) {
                const raster = rasterizeImage(rasterSource, size);
                svgImages.lod[size][name] = raster;
                rasterSource = raster;
            }
        }
    );

    await Promise.all(tasks);

    // console.log("All SVGs + LODs loaded:", svgImages);
    return svgImages;
}

const svgImages = await loadAllSvgs();
reportLoaded('pieces');

export function chooseLOD(pixelSize) {
    if (pixelSize >= 24) return svgImages.img;
    if (pixelSize >= 12) return svgImages.lod[32];
    if (pixelSize >= 6)  return svgImages.lod[16];
    if (pixelSize >= 3)  return svgImages.lod[8];
    if (pixelSize >= 2)  return svgImages.lod[4];
    if (pixelSize >= 1)  return svgImages.lod[2];
    return svgImages.lod[1];
}
