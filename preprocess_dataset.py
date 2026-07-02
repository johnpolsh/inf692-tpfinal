import shutil
import subprocess
from tqdm import tqdm
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from PIL import Image
from scipy.ndimage import distance_transform_edt


import cv2
import numpy as np
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import cg, spsolve


def timestamp(path: Path) -> float:
    """
    Extrai o timestamp do nome do arquivo.

    Exemplos:
        d-1294886362.123456-123.pgm
        r-1294886362.125987-456.ppm
    """
    return float(path.stem.split("-")[1])


def sync_frames(scene_dir: Path) -> list[tuple[float, Path, Path]]:
    """
    Associa cada frame de profundidade ao frame RGB temporalmente
    mais próximo.

    A implementação segue a mesma estratégia do get_synched_frames.m:
    - depth é a sequência primária;
    - o ponteiro do RGB nunca retrocede (complexidade O(N));
    - todos os depths são preservados.

    Retorna:
        [
            (depth_path, rgb_path),
            ...
        ]
    """

    depth_frames = sorted(scene_dir.glob("d-*"), key=timestamp)
    rgb_frames = sorted(scene_dir.glob("r-*"), key=timestamp)

    if not depth_frames:
        raise RuntimeError(f"Nenhum frame de profundidade encontrado em '{scene_dir}'.")

    if not rgb_frames:
        raise RuntimeError(f"Nenhum frame RGB encontrado em '{scene_dir}'.")

    rgb_idx = 0
    synced: list[tuple[float, Path, Path]] = []

    for depth in depth_frames:
        t_depth = timestamp(depth)

        while rgb_idx + 1 < len(rgb_frames):
            current_diff = abs(t_depth - timestamp(rgb_frames[rgb_idx]))
            next_diff = abs(t_depth - timestamp(rgb_frames[rgb_idx + 1]))

            if next_diff > current_diff:
                break

            rgb_idx += 1

        synced.append((
            t_depth,
            depth,
            rgb_frames[rgb_idx],
        ))

    return synced


def load_rgb(path: Path) -> Image.Image:
    """
    Carrega completamente uma imagem RGB.

    Lança exceção caso o arquivo esteja corrompido.
    """

    img = Image.open(path)
    img.load()

    if img.mode != "RGB":
        img = img.convert("RGB")

    return img


def process_rgb(img: Image.Image) -> Image.Image:
    """
    Espaço reservado para futuros processamentos.

    No pipeline original do NYUv2 não há nenhuma modificação
    da imagem RGB além da compressão em PNG.
    """

    return img


def save_rgb(
    img: np.ndarray,
    path: Path,
) -> None:

    img_pil = Image.fromarray(
        np.clip(
            np.round(img * 255.0),
            0,
            255,
        ).astype(np.uint8)
    )
    img_pil.save(
        path,
        format="PNG",
        optimize=True,
        compress_level=9,
    )


KINECT_INVALID = 2047

_A = 3.3309495161
_B = -0.0030711016


def load_depth(path: Path) -> np.ndarray:
    """
    Carrega um mapa de profundidade bruto do Kinect.

    Retorna um ndarray uint16 contendo a disparidade original.
    """

    depth = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if depth is None:
        raise IOError(f"Não foi possível carregar '{path}'.")

    if depth.ndim != 2:
        raise ValueError(f"'{path}' não é uma imagem monocromática.")

    depth = depth.byteswap()
    return depth


def decode_depth(raw_depth: np.ndarray) -> np.ndarray:
    """
    Converte a disparidade bruta do Kinect para profundidade em metros.

    Os pixels inválidos permanecem com valor 0.
    """

    raw_depth = raw_depth.astype(np.float32)

    depth = np.zeros_like(raw_depth, dtype=np.float32)

    valid = (
        (raw_depth > 0)
        & (raw_depth < KINECT_INVALID)
    )

    depth[valid] = 1.0 / (
        _A + _B * raw_depth[valid]
    )

    return depth


def save_depth(
    depth: np.ndarray,
    path: Path,
) -> None:
    """
    Salva um mapa de profundidade em metros como PNG de 16 bits.

    A profundidade é convertida para milímetros antes de ser salva.

    Parameters
    ----------
    depth
        ndarray float32 contendo profundidade em metros.

    path
        Caminho do PNG de saída.
    """

    depth_mm = np.clip(
        np.round(depth * 1000.0),
        0,
        np.iinfo(np.uint16).max,
    ).astype(np.uint16)

    if not cv2.imwrite(
        str(path),
        depth_mm,
        [cv2.IMWRITE_PNG_COMPRESSION, 9],
    ):
        raise IOError(f"Não foi possível salvar '{path}'.")


def _nearest_neighbor_fill(
    depth: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Preenche os pixels inválidos utilizando o pixel válido mais próximo.
    """

    _, nearest = distance_transform_edt(
        ~valid_mask,
        return_indices=True,
    )

    return depth[nearest[0], nearest[1]]


def fill_depth_colorization(
    depth: np.ndarray,
    rgb: np.ndarray,
    *,
    alpha: float = 1.0,
    radius: int = 1,
) -> np.ndarray:
    """
    Preenche o mapa de profundidade utilizando o algoritmo de colorização
    de Levin et al., adaptado pelo NYUv2.

    Parameters
    ----------
    depth
        Profundidade em metros (float32).

    rgb
        RGB float32 HxWx3 no intervalo [0,1].

    alpha
        Peso da restrição dos pixels conhecidos.

    radius
        Raio da vizinhança (o algoritmo original utiliza 1).

    Returns
    -------
    ndarray float32
    """

    H, W = depth.shape
    N = H * W

    #
    # Máscara de pixels conhecidos
    #

    known = depth > 0

    if known.all():
        return depth

    #
    # Normaliza profundidade
    #

    max_depth = depth[known].max()

    d = depth / max_depth

    #
    # Intensidade (grayscale)
    #

    gray = (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    ).astype(np.float32)

    #
    # Índice linear dos pixels
    #

    indices = np.arange(N).reshape(H, W)

    rows = []
    cols = []
    vals = []

    #
    # Termos diagonais
    #

    diag = np.ones(N, dtype=np.float32)

    #
    # Todos os deslocamentos da janela
    #

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):

            if dx == 0 and dy == 0:
                continue

            #
            # Fatias válidas
            #

            y0 = max(0, dy)
            y1 = H + min(0, dy)

            x0 = max(0, dx)
            x1 = W + min(0, dx)

            center = indices[y0:y1, x0:x1]
            neigh = indices[
                y0 - dy:y1 - dy,
                x0 - dx:x1 - dx,
            ]

            Ic = gray[y0:y1, x0:x1]
            In = gray[
                y0 - dy:y1 - dy,
                x0 - dx:x1 - dx,
            ]

            diff = In - Ic

            #
            # Variância local (aproximação)
            #

            sigma = np.maximum(
                0.6 * diff**2,
                2e-6,
            )

            weight = np.exp(-(diff**2) / sigma)

            rows.append(center.ravel())
            cols.append(neigh.ravel())
            vals.append(-weight.ravel())

            diag[center.ravel()] += weight.ravel()

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)

    #
    # Matriz de suavidade
    #

    A = csr_matrix(
        (vals, (rows, cols)),
        shape=(N, N),
    )

    A = A + diags(diag)

    #
    # Restrições
    #

    G = diags(
        known.astype(np.float32).ravel() * alpha
    )

    b = (
        known.astype(np.float32).ravel()
        * alpha
        * d.ravel()
    )

    M = A + G

    #
    # Resolve sistema
    #

    x, info = cg(
        M,
        b,
        atol=1e-6,
        maxiter=500,
    )

    if info != 0:
        x = spsolve(M, b)

    result = x.reshape(H, W)

    result *= max_depth

    #
    # Preserva exatamente os pixels válidos
    #

    result[known] = depth[known]

    return result.astype(np.float32)


def fill_depth(
    depth: np.ndarray,
    rgb: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    morph_iters: int = 3,
    guided_radius: int = 4,
    guided_eps: float = 1e-2,
    min_depth: float = 0.1,
) -> np.ndarray:
    """
    Preenche regiões sem profundidade utilizando:

        1. Dilatação morfológica progressiva;
        2. Nearest-neighbor para buracos remanescentes;
        3. Guided Filter utilizando a imagem RGB.

    Parameters
    ----------
    depth
        Profundidade em metros (float32).

    rgb
        RGB H×W×3 em float32 no intervalo [0, 1].

    valid_mask
        Máscara opcional dos pixels válidos.
        Caso não seja informada, considera depth > 0.

    Returns
    -------
    ndarray float32
    """

    if valid_mask is None:
        valid_mask = depth > 0

    filled = depth.copy()
    filled[~valid_mask] = 0

    current_mask = valid_mask.copy()

    #
    # Dilatação progressiva
    #

    for i in range(morph_iters):

        kernel_size = 3

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        dilated = cv2.dilate(filled, kernel)

        update = (~current_mask) & (dilated > 0)

        filled[update] = dilated[update]

        current_mask |= update

        if current_mask.all():
            break

    #
    # Nearest neighbor para buracos restantes
    #

    if not current_mask.all():
        filled = _nearest_neighbor_fill(
            filled,
            current_mask,
        )

    #
    # Guided Filter
    #

    luminance = (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    ).astype(np.float32)

    filled = cv2.ximgproc.guidedFilter(
        guide=luminance,
        src=filled,
        radius=guided_radius,
        eps=guided_eps,
    )

    #
    # Mantém exatamente os valores medidos pelo Kinect
    #

    filled[valid_mask] = depth[valid_mask]

    return np.maximum(filled, min_depth).astype(np.float32)


FX_RGB = 518.8579011745019
FY_RGB = 519.4696111212748
CX_RGB = 325.58244941119034
CY_RGB = 253.73616633400465

FX_D = 582.6244816773795
FY_D = 582.6910327098864
CX_D = 313.0447587080473
CY_D = 238.44389626620386

DIST_RGB = np.array([
    2.0796615318809061e-01,
   -5.8613825163911781e-01,
    7.2231363135888329e-04,
    1.0479627195765181e-03,
    4.9856986684705107e-01,
])

DIST_D = np.array([
   -9.9897236553084481e-02,
    3.9065324602765344e-01,
    1.9290592870229277e-03,
   -1.9422022475975055e-03,
   -5.1031725053400578e-01,
])

K_RGB = np.array([
    [FX_RGB, 0, CX_RGB],
    [0, FY_RGB, CY_RGB],
    [0, 0, 1],
], np.float64)

K_DEPTH = np.array([
    [FX_D, 0, CX_D],
    [0, FY_D, CY_D],
    [0, 0, 1],
], np.float64)

R = np.array([
    [ 9.9997798940829263e-01,  5.0518419386157446e-03,  4.3011152014118693e-03],
    [-5.0359919480810989e-03,  9.9998051861143999e-01, -3.6879781309514218e-03],
    [-4.3196624923060242e-03,  3.6662365748484798e-03,  9.9998394948385538e-01],
])

R = np.linalg.inv(R.T)

T = np.array([
    2.5031875059141302e-02,
    6.6238747008330102e-04,
   -2.9342312935846411e-04,
], np.float64)

H = 480
W = 640

RGB_MAP1, RGB_MAP2 = cv2.initUndistortRectifyMap(
    K_RGB,
    DIST_RGB,
    None,
    K_RGB,
    (W, H),
    cv2.CV_32FC1,
)


def warp_rgb_to_depth(
    rgb: np.ndarray,
    depth: np.ndarray,
) -> np.ndarray:
    """
    Reprojeta a imagem RGB para o referencial da câmera de profundidade.

    Parameters
    ----------
    rgb
        Imagem RGB H×W×3 em float32 no intervalo [0,1].

    depth
        Profundidade em metros (HxW).

    Returns
    -------
    rgb_depth
        RGB alinhada ao depth.
    """

    H, W = depth.shape

    #
    # Grade de pixels do depth
    #

    u_d, v_d = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
    )

    valid = depth > 0

    z = depth[valid]
    u_d = u_d[valid]
    v_d = v_d[valid]

    #
    # Backprojection na câmera de profundidade
    #

    X = (u_d - CX_D) * z / FX_D
    Y = (v_d - CY_D) * z / FY_D

    pts_depth = np.stack(
        (X, Y, z),
        axis=0,
    )

    #
    # Transformação para a câmera RGB
    #

    pts_rgb = R.T @ (pts_depth - T[:, None])

    Xr = pts_rgb[0]
    Yr = pts_rgb[1]
    Zr = pts_rgb[2]

    #
    # Remove pontos inválidos
    #

    valid = Zr > 0

    Xr = Xr[valid]
    Yr = Yr[valid]
    Zr = Zr[valid]

    u_d = u_d[valid]
    v_d = v_d[valid]

    #
    # Projeção na câmera RGB
    #

    x = Xr / Zr
    y = Yr / Zr

    r2 = x * x + y * y

    radial = (
        1
        + DIST_RGB[0] * r2
        + DIST_RGB[1] * r2**2
        + DIST_RGB[4] * r2**3
    )

    x_dist = (
        x * radial
        + 2 * DIST_RGB[2] * x * y
        + DIST_RGB[3] * (r2 + 2 * x * x)
    )

    y_dist = (
        y * radial
        + DIST_RGB[2] * (r2 + 2 * y * y)
        + 2 * DIST_RGB[3] * x * y
    )

    u_rgb = FX_RGB * x_dist + CX_RGB
    v_rgb = FY_RGB * y_dist + CY_RGB

    #
    # Mapas para remap
    #

    map_x = np.full(
        (H, W),
        -1,
        dtype=np.float32,
    )

    map_y = np.full(
        (H, W),
        -1,
        dtype=np.float32,
    )

    map_x[
        v_d.astype(np.int32),
        u_d.astype(np.int32),
    ] = u_rgb.astype(np.float32)

    map_y[
        v_d.astype(np.int32),
        u_d.astype(np.int32),
    ] = v_rgb.astype(np.float32)

    #
    # Preenche pixels sem correspondência
    #

    invalid = map_x < 0

    if invalid.any():

        _, nearest = distance_transform_edt(
            invalid,
            return_indices=True,
        )

        map_x = map_x[
            nearest[0],
            nearest[1],
        ]

        map_y = map_y[
            nearest[0],
            nearest[1],
        ]

    #
    # Warp
    #

    warped = cv2.remap(
        rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )

    return warped


def undistort_rgb(rgb: np.ndarray) -> np.ndarray:
    return cv2.remap(
        rgb,
        RGB_MAP1,
        RGB_MAP2,
        interpolation=cv2.INTER_LINEAR,
    )


def process_scene(
    scene_dir: Path,
    interval: float = 0.5,
) -> None:
    """
    Processa uma cena do NYUv2.

    Para cada intervalo de tempo:
      - seleciona o primeiro frame sincronizado após o instante alvo;
      - tenta carregar RGB e Depth;
      - caso algum esteja corrompido, tenta o próximo;
      - processa ambos;
      - salva como PNG;
      - remove os arquivos originais ao final.
    """

    pairs = sync_frames(scene_dir)

    if not pairs:
        return

    used_files: set[Path] = set()

    target_time = pairs[0][0]
    index = 0

    while index < len(pairs):

        #
        # procura o primeiro frame >= target_time
        #

        while (
            index < len(pairs)
            and pairs[index][0] < target_time
        ):
            index += 1

        if index >= len(pairs):
            break

        #
        # procura o primeiro par válido
        #

        while index < len(pairs):

            timestamp, depth_path, rgb_path = pairs[index]

            try:

                #
                # carregamento (já valida os arquivos)
                #

                rgb = load_rgb(rgb_path)
                depth = load_depth(depth_path)

            except (
                OSError,
                ValueError,
                IOError,
                UnidentifiedImageError,
            ):
                index += 1
                continue

            #
            # processamento RGB
            #

            rgb = process_rgb(rgb)

            rgb_np = np.asarray(rgb, dtype=np.float32) / 255.0

            depth = decode_depth(depth)

            depth = fill_depth(
                depth,
                rgb_np,
            )

            rgb = undistort_rgb(rgb_np)
            rgb = warp_rgb_to_depth(
                rgb,
                depth,
            )

            #
            # salvamento
            #

            rgb_png = rgb_path.with_suffix(".png")
            depth_png = depth_path.with_suffix(".png")

            save_rgb(rgb, rgb_png)
            save_depth(depth, depth_png)

            used_files.add(rgb_png)
            used_files.add(depth_png)

            target_time = timestamp + interval

            index += 1

            break

    #
    # remove arquivos antigos
    #

    for path in scene_dir.iterdir():
        if path in used_files:
            continue

        if path.name.startswith(("r-", "d-")):
            path.unlink()


def process(extracted_dirs: list[Path]) -> None:
    """
    Processa todas as cenas extraídas de um arquivo ZIP.

    Cada subdiretório representa uma cena do NYUv2.
    """

    total = len(extracted_dirs)
    for i, scene_dir in tqdm(enumerate(sorted(extracted_dirs), start=1), total=total):

        if not scene_dir.is_dir():
            continue

        try:
            process_scene(scene_dir)

        except KeyboardInterrupt:
            raise

        except Exception as e:
            print(f"Erro ao processar '{scene_dir.name}': {e}")

    #
    # Remove diretórios vazios (opcional)
    #

    for scene_dir in extracted_dirs:
        try:
            next(scene_dir.iterdir())
        except StopIteration:
            shutil.rmtree(scene_dir)


import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse


def extract_zips(
    root: str | Path,
    urls_file: str | Path = "urls.txt",
    output_dir: str | Path = "nyu_depth_v2",
) -> None:

    root = Path(root)
    output_dir = root / output_dir
    urls_file = root / urls_file

    output_dir.mkdir(parents=True, exist_ok=True)

    with urls_file.open() as f:
        urls = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

    for url in urls:

        filename = Path(urlparse(url).path).name
        zip_path = root / filename

        print(f"Baixando {filename}")

        subprocess.run(
            [
                "wget",
                "-O",
                str(zip_path),
                url,
            ],
            check=True,
        )

        print(f"Extraindo {filename}")

        before = {
            p.name
            for p in output_dir.iterdir()
            if p.is_dir()
        }

        subprocess.run(
            [
                "unzip",
                "-o",
                str(zip_path),
                "-x",
                "*/a-*.dump",
                "-d",
                str(output_dir),
            ],
            check=True,
        )

        extracted_dirs = sorted(
            p
            for p in output_dir.iterdir()
            if p.is_dir() and p.name not in before
        )

        process(extracted_dirs)

        print(f"Removendo {filename}")

        zip_path.unlink(missing_ok=True)


extract_zips(
    "./nyu_depth_v2/",
    output_dir="preprocessed"
)
