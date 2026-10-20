"""
Vectorized Culture Construction
================================

This module replaces the slow nested-loop culture construction with
fully vectorized numpy operations.

Expected speedup: 50-200x depending on lado and n values.

Key optimizations:
1. Broadcasting instead of loops
2. Pre-computed index arrays
3. Efficient boolean operations for neighbor detection
"""
#from enum import IntEnum
#class SystemConst(IntEnum):
#    # A nivel pixelar
#    EMPTY = 0
#    MEMBRANE = 1
#    TJ = 8
#    MEM_FUS = 4
#    TJ_FUS = 8
#    ELECTRODE = 1
#    # A nivel Celular
#    ALIVE     = 1
#    DEAD      = 0
#    FUSED     = 2

from ConstantsFile import SystemConst

import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import label as _scipy_label
from matplotlib.path import Path

def construct_substrate_vectorized(
    side: int,        # Height and width del square (píxeles)
    de: float,        # Electrode diameter [cm]
    d: float          # Tamaño de píxel [cm]
) -> np.ndarray:
    """
    Substrate construction matching hexagonal culture dimensions.
    
    Args:
        h, w: Dimensiones del square en píxeles
        de: Electrode diameter [cm]
        d: Pixel size [cm] (ac / membrane_per_cell)
    
    Returns:
        Substrate_vector: Flattened array with 1=electrode, 0=insulator
    """
    # Coordenadas físicas de cada píxel
    x_coords = np.arange(side) * d
    y_coords = np.arange(side) * d
    
    # Centro del electrodo (centro del square)
    x_center = side * d / 2
    y_center = side * d / 2
    
    # Distancia de cada píxel al centro
    X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')
    dist_sq = (X - x_center)**2 + (Y - y_center)**2
    
    # Región del electrodo
    electrode_radius = de / 2
    Substrate = SystemConst.ELECTRODE*(dist_sq <= electrode_radius**2).astype(np.int8)
    
    # Flatten en mismo orden que culture
    return Substrate.ravel(order='F')

class HexCell:
    """
    Stencil de referencia de una célula hexagonal centrada en (0,0).
    Se usa como plantilla para todo el cultivo.
    """
    def __init__(self, resolution: int, height_width_ratio: float = 1.0):
        """
        Args:
            resolution: Píxeles por unidad (radio base)
            height_width_ratio: Ratio alto/ancho (1.0 = regular, >1 = alto, <1 = ancho)
        """
        self.resolution = resolution
        self.ratio = height_width_ratio

        # Dimensiones en píxeles — redondear al múltiplo-de-4 más cercano para
        # evitar truncamiento en cadena:
        #   height_px par  => h = H/2 entero exacto  => spacing_y = H sin error
        #   width_px ≡ 0 mod 4  => R = w/2 par entero => R/2 entero exacto en vértices
        #self.width_px  = round(2 * resolution / np.sqrt(self.ratio) / 4) * 4
        #self.height_px = round(resolution * np.sqrt(3) * np.sqrt(self.ratio) / 2) * 2
        
        self.width_px  = int(resolution / np.sqrt(self.ratio))
        self.height_px = int(resolution * np.sqrt(3) * np.sqrt(self.ratio)/2)
        if self.width_px % 2 != 0: self.width_px += 1
        if self.height_px % 2 != 0: self.height_px += 1

        if self.width_px == 0 or self.height_px == 0:
            raise ValueError('The geometrical figure must not be deformed that much. Use a different HW ratio.')
        # Vértices relativos a centro (0,0)
        self.vertices = self._compute_vertices()
        
        # Cache de offsets
        self._tj_offsets = None
        self._membrane_offsets = None
        self._all_offsets = None

    def _compute_vertices(self):
        """Calcula vértices del hexágono flat-top centrado en (0,0)."""
        R = self.width_px  / 2
        h = self.height_px / 2
        
        # Orden CCW empezando por derecha
        return np.array([
            [R,    0],      # 0: derecha
            [R/2,  h],      # 1: arriba-derecha
            [-R/2, h],      # 2: arriba-izquierda
            [-R,   0],      # 3: izquierda
            [-R/2, -h],     # 4: abajo-izquierda
            [R/2,  -h],     # 5: abajo-derecha
        ], dtype=np.int32)
    
    def centers(self, n_cols: int, n_rows: int) -> np.ndarray:
        cols = np.arange(n_cols)
        rows = np.arange(n_rows)

        col_grid, row_grid = np.meshgrid(cols, rows, indexing='ij')

        x = col_grid * self.spacing_x
        y = row_grid * self.spacing_y

        # offset en columnas impares
        y += (col_grid & 1) * (self.spacing_y // 2)

        centers = np.stack([x, y], axis=-1)
        return centers.reshape(-1, 2).astype(np.int32)

    def cell_index(self, col: int, row: int, n_rows: int) -> int:
        return col * n_rows + row

    def neighbors(self, n_cols: int, n_rows: int) -> list[list[int]]:
        neighbors = [[] for _ in range(n_cols * n_rows)]

        for col in range(n_cols):
            for row in range(n_rows):
                idx = self.cell_index(col, row, n_rows)

                if col & 1 == 0:
                    deltas = [(0,-1),(0,1),(-1,0),(-1,-1),(1,0),(1,-1)]
                else:
                    deltas = [(0,-1),(0,1),(-1,0),(-1,1),(1,0),(1,1)]

                for dc, dr in deltas:
                    c2, r2 = col + dc, row + dr
                    if 0 <= c2 < n_cols and 0 <= r2 < n_rows:
                        neighbors[idx].append(
                            self.cell_index(c2, r2, n_rows)
                        )
        return neighbors

    @property
    def spacing_x(self):
        """Distancia horizontal entre centros para solapar vértices [1]-[4]."""
        return int(self.width_px * 3 / 4)
    
    @property
    def spacing_y(self):
        """Distancia vertical entre centros para solapar vértices [1,2]-[4,5]."""
        return self.vertices[1][1] - self.vertices[4][1]  # top_y - bottom_y
    
    def tj_offsets(self):
        """Retorna offsets (dy, dx) de píxeles TJ relativos a centro (0,0)."""
        if self._tj_offsets is None:
            self._compute_all_offsets()
        return self._tj_offsets
    
    def membrane_offsets(self):
        """Retorna offsets (dy, dx) de píxeles membrana relativos a centro (0,0)."""
        if self._membrane_offsets is None:
            self._compute_all_offsets()
        return self._membrane_offsets
    
    def all_offsets(self):
        """Retorna offsets (dy, dx) de todos los píxeles de la célula."""
        if self._all_offsets is None:
            self._compute_all_offsets()
        return self._all_offsets
    
    def _compute_all_offsets(self):
        """Calcula offsets de TJ y membrana de forma eficiente."""
        v = self.vertices
        
        # 1. Calcular TJ (borde del hexágono)
        v_next = np.roll(v, -1, axis=0)
        diffs = np.abs(v_next - v)
        lengths = np.maximum(diffs[:, 0], diffs[:, 1]) + 1
        total_tj = lengths.sum()
        
        tj_x = np.empty(total_tj, dtype=np.int32)
        tj_y = np.empty(total_tj, dtype=np.int32)
        
        pos = 0
        for i in range(len(v)):
            n = lengths[i]
            t = np.linspace(0, 1, n)
            tj_x[pos:pos+n] = np.round(v[i, 0] + t * (v_next[i, 0] - v[i, 0]))
            tj_y[pos:pos+n] = np.round(v[i, 1] + t * (v_next[i, 1] - v[i, 1]))
            pos += n
        
        tj_pixels = np.column_stack([tj_y, tj_x])
        _, idx = np.unique(tj_pixels, axis=0, return_index=True)
        self._tj_offsets = tj_pixels[np.sort(idx)]
        
        # 2. Calcular membrana (interior)
        path = Path(v)
        
        x_min, x_max = v[:, 0].min(), v[:, 0].max()
        y_min, y_max = v[:, 1].min(), v[:, 1].max()
        
        x_range = np.arange(x_min, x_max + 1)
        y_range = np.arange(y_min, y_max + 1)
        xx, yy = np.meshgrid(x_range, y_range)
        points = np.column_stack([xx.ravel(), yy.ravel()])
        
        inside = path.contains_points(points)
        interior_xy = points[inside]
        interior_yx = interior_xy[:, ::-1]  # (x,y) → (y,x)
        
        # Remover TJ del interior
        tj_set = set(map(tuple, self._tj_offsets))
        self._membrane_offsets = np.array([p for p in interior_yx if tuple(p) not in tj_set])
        
        # 3. Todos los píxeles
        self._all_offsets = np.vstack([self._membrane_offsets, self._tj_offsets])
    
    def tj_offsets_no_vertices(self):
        """TJ sin incluir vértices del hexágono"""
        if self._tj_offsets is None:
            self._compute_all_offsets()
        
        tj = self._tj_offsets                  # (N, 2) (dy, dx)
        verts = self.vertices[:, ::-1]         # (6, 2) (dy, dx)

        # Comparación fila-a-fila con broadcasting
        is_vertex = np.any(
            np.all(tj[:, None, :] == verts[None, :, :], axis=2),
            axis=1
        )

        tj_off_nv = tj[~is_vertex]
                
        return tj_off_nv
    
    def get_offsets(self, row : int = 0 ):
        """Siempre retorna los mismos offsets."""
        return self.membrane_offsets(), self.tj_offsets() , self.all_offsets()
    
    def pixel_count(self):
        """Retorna cantidad total de píxeles en la célula."""
        return len(self.all_offsets())
    
    def tj_count(self):
        """Retorna cantidad de píxeles TJ."""
        return len(self.tj_offsets())
    
    def membrane_count(self):
        """Retorna cantidad de píxeles de membrana."""
        return len(self.membrane_offsets())

class RecCell:
    """
    Stencil de referencia de una célula Rectangular centrada en (0,0).
    Se usa como plantilla para todo el cultivo.
    """
    def __init__(self, resolution: int, height_width_ratio: float = 1.0, displacement: float = 0.0):
            """
            Args:
                resolution: equal to the base width and height of the square. If it is disorted: width = res/sqrt(ratio) & height = res * sqrt(ratio).
                height_width_ratio: Ratio height/width (1.0 = regular, >1 = higher, <1 = widther).
                displacement: offset between even and odd collumns. Goes from -.5 to .5 cell size. More of that is symmetric.
            """
            if not (-0.5 <= displacement <= 0.5):
                print("Displacement must be between -0.5 and 0.5. it will be looped by symetry.")
                displacement = ((displacement + 0.5) % 1.0) - 0.5
            
            self.displacement = displacement
            self.resolution = resolution
            self.ratio = height_width_ratio
            # Dimensiones en píxeles
            self.width_px  = int(resolution / np.sqrt(self.ratio))
            self.height_px = int(resolution * np.sqrt(self.ratio))

            if self.width_px == 0: 
                print("The cell was too disorted, the width and height will be fixed to the limit of the selected resolution.")
                self.width_px  = 2
                self.height_px = resolution//2
            elif self.height_px == 0:
                print("The cell was too disorted, the width and height will be fixed to the limit of the selected resolution.")
                self.height_px = 2
                self.width_px = resolution//2

            if self.width_px % 2 == 1  : self.width_px  += 1
            elif self.width_px == 0: self.width_px  += 1
            if self.height_px % 2 == 1 : self.height_px += 1
            elif self.height_px == 0: self.height_px  += 1
            

            # Vértices relativos a centro (0,0)
            self.vertices = self._compute_vertices()
            
            # Cache de offsets
            self._tj_offsets = None
            self._membrane_offsets = None
            self._all_offsets = None

    def _compute_vertices(self):
        """Calcula vértices del hexágono flat-top centrado en (0,0)."""
        R = self.width_px  / 2
        h = self.height_px / 2
        
        # Orden CCW empezando por derecha
        return np.array([
            [R,  h],      # 1: arriba-derecha
            [-R, h],      # 2: arriba-izquierda
            [-R, -h],     # 4: abajo-izquierda
            [R,  -h],     # 5: abajo-derecha
        ], dtype=np.int32)
    
    def centers(self, n_cols: int, n_rows: int) -> np.ndarray:
        cols = np.arange(n_cols)
        rows = np.arange(n_rows)

        col_grid, row_grid = np.meshgrid(cols, rows, indexing='ij')

        x = col_grid * self.spacing_x
        y = row_grid * self.spacing_y

        # offset en columnas impares
        y += (col_grid & 1) * int(self.height_px * self.displacement)

        centers = np.stack([x, y], axis=-1)
        return centers.reshape(-1, 2).astype(np.int32)

    def cell_index(self, col: int, row: int, n_rows: int) -> int:
        return col * n_rows + row

    def neighbors(self, n_cols: int, n_rows: int) -> list[list[int]]:
        neighbors = [[] for _ in range(n_cols * n_rows)]

        for col in range(n_cols):
            for row in range(n_rows):
                idx = self.cell_index(col, row, n_rows)

                if self.displacement == 0:
                    deltas = [(0,-1),(0,1),(-1,0),(1,0)]
                elif self.displacement > 0:
                    # odd columns shifted DOWN → brick-wall topology, 6 neighbors
                    if col & 1 == 0:
                        deltas = [(0,-1),(0,1),(-1,-1),(1,-1),(-1,0),(1,0)]
                    else:
                        deltas = [(0,-1),(0,1),(-1,0),(1,0),(-1,1),(1,1)]
                else:
                    # odd columns shifted UP, mirror of displacement > 0
                    if col & 1 == 0:
                        deltas = [(0,-1),(0,1),(-1,0),(1,0),(-1,1),(1,1)]
                    else:
                        deltas = [(0,-1),(0,1),(-1,-1),(1,-1),(-1,0),(1,0)]

                for dc, dr in deltas:
                    c2, r2 = col + dc, row + dr
                    if 0 <= c2 < n_cols and 0 <= r2 < n_rows:
                        neighbors[idx].append(
                            self.cell_index(c2, r2, n_rows)
                        )

        return neighbors

    @property
    def spacing_x(self):
        """Distancia horizontal entre centros para solapar vértices [1]-[4]."""
        return int(self.width_px)
    @property
    def spacing_y(self):
        return int(self.height_px)  # top_y - bottom_y
    
    def tj_offsets(self):
        """Retorna offsets (dy, dx) de píxeles TJ relativos a centro (0,0)."""
        if self._tj_offsets is None:
            self._compute_all_offsets()
        return self._tj_offsets
    
    def membrane_offsets(self):
        """Retorna offsets (dy, dx) de píxeles membrana relativos a centro (0,0)."""
        if self._membrane_offsets is None:
            self._compute_all_offsets()
        return self._membrane_offsets
    
    def all_offsets(self):
        """Retorna offsets (dy, dx) de todos los píxeles de la célula."""
        if self._all_offsets is None:
            self._compute_all_offsets()
        return self._all_offsets
    
    def _compute_all_offsets(self):
        """Calcula offsets de TJ y membrana de forma eficiente."""
        v = self.vertices
        # 1. Calcular TJ (borde del hexágono)
        v_next = np.roll(v, -1, axis=0)
        diffs = np.abs(v_next - v)
        lengths = np.maximum(diffs[:, 0], diffs[:, 1]) + 1
        total_tj = lengths.sum()
        if total_tj == 0:
            self._tj_offsets = np.empty((0, 2), dtype=np.int32)
            self._membrane_offsets = np.empty((0, 2), dtype=np.int32)
            self._all_offsets = np.vstack([self._membrane_offsets, self._tj_offsets])
            return
        
        tj_x = np.empty(total_tj, dtype=np.int32)
        tj_y = np.empty(total_tj, dtype=np.int32)
        
        pos = 0
        for i in range(len(v)):
            n = lengths[i]
            if n > 0:
                t = np.linspace(0, 1, n)
                tj_x[pos:pos+n] = np.round(v[i, 0] + t * (v_next[i, 0] - v[i, 0]))
                tj_y[pos:pos+n] = np.round(v[i, 1] + t * (v_next[i, 1] - v[i, 1]))
                pos += n
        
        tj_pixels = np.column_stack([tj_y, tj_x])
        if len(tj_pixels) > 0:
            _, idx = np.unique(tj_pixels, axis=0, return_index=True)
            self._tj_offsets = tj_pixels[np.sort(idx)]
        else:
            self._tj_offsets = np.empty((0, 2), dtype=np.int32)
        
        # 2. Calcular membrana (interior)
        path = Path(v.astype(float))
        x_min, x_max = int(v[:, 0].min()), int(v[:, 0].max())
        y_min, y_max = int(v[:, 1].min()), int(v[:, 1].max())
        if x_max > x_min and y_max > y_min:
            x_range = np.arange(x_min, x_max + 1)
            y_range = np.arange(y_min, y_max + 1)
            xx, yy = np.meshgrid(x_range, y_range)
            points = np.column_stack([xx.ravel(), yy.ravel()])
        
            inside = path.contains_points(points)
            interior_xy = points[inside]
            interior_yx = interior_xy[:, ::-1]  # (x,y) → (y,x)
            
            if len(interior_xy) > 0:
                interior_yx = interior_xy[:, ::-1]
                tj_set = set(map(tuple, self._tj_offsets))
                self._membrane_offsets = np.array([
                    p for p in interior_yx if tuple(p) not in tj_set
                ], dtype=np.int32)
            else:
                self._membrane_offsets = np.empty((0, 2), dtype=np.int32)

        else:
            self._membrane_offsets = np.empty((0, 2), dtype=np.int32)
        self._all_offsets = np.vstack([self._membrane_offsets, self._tj_offsets])
    
    def tj_offsets_no_vertices(self):
        """TJ sin incluir vértices del hexágono"""
        if self._tj_offsets is None:
            self._compute_all_offsets()
        
        tj = self._tj_offsets                  # (N, 2) (dy, dx)
        verts = self.vertices[:, ::-1]         # (6, 2) (dy, dx)

        # Comparación fila-a-fila con broadcasting
        is_vertex = np.any(
            np.all(tj[:, None, :] == verts[None, :, :], axis=2),
            axis=1
        )
        return tj[~is_vertex]
 
    def get_offsets(self, row : int = 0):
        return self.membrane_offsets(), self.tj_offsets(), self.all_offsets()
    
    def pixel_count(self):
        """Retorna cantidad total de píxeles en la célula."""
        return len(self.all_offsets())
    
    def tj_count(self):
        """Retorna cantidad de píxeles TJ."""
        return len(self.tj_offsets())
    
    def membrane_count(self):
        """Retorna cantidad de píxeles de membrana."""
        return len(self.membrane_offsets())

class TriCell:
    """
    Stencil de referencia de una célula Triangular centrada en (0,0).
    Se usa como plantilla para todo el cultivo.
    """
    def __init__(self, resolution: int, height_width_ratio: float = 1.0):
            """
            Args:
                resolution: equal to the base width and height of the square. If it is disorted: width = res/sqrt(ratio) & height = res * sqrt(ratio).
                height_width_ratio: Ratio height/width (1.0 = regular, >1 = higher, <1 = widther).
            """        

            self.resolution = resolution
            self.ratio = height_width_ratio
            # Dimensiones en píxeles
            self.width_px = int(resolution / np.sqrt(self.ratio))
            self.height_px = int(resolution * np.sqrt(3) * np.sqrt(self.ratio)/2)
            
            if self.width_px == 0 or self.height_px == 0:
                raise ValueError('The geometrical figure must not be deformed that much. Use a different HW ratio or a bigger resolution.')
            # Ajustar a pares para centros enteros
            if self.width_px % 2 != 0:
                self.width_px += 1
            if self.height_px % 3 != 0:
                self.height_px += 3-self.height_px % 3
            
            # Vértices relativos a centro (0,0)
            self.vertices = np.zeros(3) 
            self.vertices_down = np.zeros(3)
            self._compute_vertices()
            
            # Cache de offsets
            self._tj_offsets = None
            self._membrane_offsets = None
            self._all_offsets = None
            self._tj_offsets_down = None
            self._membrane_offsets_down = None
            self._all_offsets_down = None

    def _compute_vertices(self):
        """Calcula vértices del hexágono flat-top centrado en (0,0)."""
        w = self.width_px  // 2
        h = self.height_px // 3

        # Punta arriba: centroid está 1/3 desde la base, 2/3 desde el ápice
        self.vertices = np.array([
                [0, 2*h],      # Ápice (arriba)
                [-w, -h],      # Base izquierda
                [w, -h],       # Base derecha
            ], dtype=np.int32)
        self.vertices_down = np.array([
                [0, -2*h],     # Ápice (abajo)
                [-w, h],       # Base izquierda (arriba)
                [w, h],        # Base derecha (arriba)
            ], dtype=np.int32)
    
    def centers(self, n_cols: int, n_rows: int) -> np.ndarray:
        cols = np.arange(n_cols)
        rows = np.arange(n_rows)
        col_grid, row_grid = np.meshgrid(cols, rows, indexing='ij')

        x = col_grid * self.spacing_x
        y = row_grid * self.spacing_y

        # Offset en Y para triángulos alternados (1/3 del spacing_y)
        y += ((col_grid + row_grid) % 2 == 1) * (self.spacing_y // 3)
        
        centers = np.stack([x, y], axis=-1)
        return centers.reshape(-1, 2).astype(np.int32)

    def cell_index(self, col: int, row: int, n_rows: int) -> int:
        return col * n_rows + row

    def neighbors(self, n_cols: int, n_rows: int) -> list[list[int]]:
        neighbors = [[] for _ in range(n_cols * n_rows)]
        for col in range(n_cols):
            for row in range(n_rows):
                idx = self.cell_index(col, row, n_rows)
                if (col + row) % 2 == 0:
                    # Pointy-up triangle: neighbors are left, right, bottom
                    deltas = [(-1,0),(1,0),(0,-1)]
                else:
                    # Pointy-down triangle: neighbors are left, right, top
                    deltas = [(-1,0),(1,0),(0,1)]
                for dc, dr in deltas:
                    c2, r2 = col + dc, row + dr
                    if 0 <= c2 < n_cols and 0 <= r2 < n_rows:
                        neighbors[idx].append(
                            self.cell_index(c2, r2, n_rows)
                        )
        return neighbors

    @property
    def spacing_x(self):
        return self.width_px//2
    @property
    def spacing_y(self):
        return self.height_px
    
    def tj_offsets(self):
        """Retorna offsets (dy, dx) de píxeles TJ relativos a centro (0,0)."""
        if self._tj_offsets is None:
            self._compute_all_offsets()
        return self._tj_offsets
    
    def membrane_offsets(self):
        """Retorna offsets (dy, dx) de píxeles membrana relativos a centro (0,0)."""
        if self._membrane_offsets is None:
            self._compute_all_offsets()
        return self._membrane_offsets
    
    def all_offsets(self):
        """Retorna offsets (dy, dx) de todos los píxeles de la célula."""
        if self._all_offsets is None:
            self._compute_all_offsets()
        return self._all_offsets
    
    def _compute_all_offsets(self):
        """Calcula offsets de TJ y membrana de forma eficiente."""
        v = self.vertices
        # 1. Calcular TJ (borde del hexágono)
        v_next = np.roll(v, -1, axis=0)
        diffs = np.abs(v_next - v)
        lengths = np.maximum(diffs[:, 0], diffs[:, 1]) + 1
        total_tj = lengths.sum()
        if total_tj == 0:
            self._tj_offsets = np.empty((0, 2), dtype=np.int32)
            self._membrane_offsets = np.empty((0, 2), dtype=np.int32)
            self._all_offsets = np.vstack([self._membrane_offsets, self._tj_offsets])
            return
        
        tj_x = np.empty(total_tj, dtype=np.int32)
        tj_y = np.empty(total_tj, dtype=np.int32)
        pos = 0
        for i in range(len(v)):
            n = lengths[i]
            if n > 0:
                t = np.linspace(0, 1, n)
                tj_x[pos:pos+n] = np.round(v[i, 0] + t * (v_next[i, 0] - v[i, 0]))
                tj_y[pos:pos+n] = np.round(v[i, 1] + t * (v_next[i, 1] - v[i, 1]))
                pos += n
        tj_pixels = np.column_stack([tj_y, tj_x])
        if len(tj_pixels) > 0:
            _, idx = np.unique(tj_pixels, axis=0, return_index=True)
            self._tj_offsets = tj_pixels[np.sort(idx)]
        else:
            self._tj_offsets = np.empty((0, 2), dtype=np.int32)
        
        # 2. Calcular membrana (interior)
        path = Path(v.astype(float))
        x_min, x_max = int(v[:, 0].min()), int(v[:, 0].max())
        y_min, y_max = int(v[:, 1].min()), int(v[:, 1].max())
        if x_max > x_min and y_max > y_min:
            x_range = np.arange(x_min, x_max + 1)
            y_range = np.arange(y_min, y_max + 1)
            xx, yy = np.meshgrid(x_range, y_range)
            points = np.column_stack([xx.ravel(), yy.ravel()])
        
            inside = path.contains_points(points)
            interior_xy = points[inside]
            interior_yx = interior_xy[:, ::-1]  # (x,y) → (y,x)
            
            if len(interior_xy) > 0:
                interior_yx = interior_xy[:, ::-1]
                tj_set = set(map(tuple, self._tj_offsets))
                self._membrane_offsets = np.array([
                    p for p in interior_yx if tuple(p) not in tj_set
                ], dtype=np.int32)
            else:
                self._membrane_offsets = np.empty((0, 2), dtype=np.int32)

        else:
            self._membrane_offsets = np.empty((0, 2), dtype=np.int32)
        self._all_offsets = np.vstack([self._membrane_offsets, self._tj_offsets])
    
    def tj_offsets_no_vertices(self):
        """TJ sin incluir vértices del hexágono"""
        if self._tj_offsets is None:
            self._compute_all_offsets()
        
        tj = self._tj_offsets                  # (N, 2) (dy, dx)
        verts = self.vertices[:, ::-1]         # (6, 2) (dy, dx)

        # Comparación fila-a-fila con broadcasting
        is_vertex = np.any(
            np.all(tj[:, None, :] == verts[None, :, :], axis=2),
            axis=1
        )
        return tj[~is_vertex]
    
    def get_offsets(self, row: int = 0):
        """
        Retorna (membrane_off, tj_off, all_off) según paridad de fila.
        Filas pares: pointy-up (normal)
        Filas impares: pointy-down (flipped en Y)
        """
        mem = self.membrane_offsets()
        tj = self.tj_offsets()
        all_off = self.all_offsets()
        
        if row % 2 == 1:
            # Flip en eje Y: (dy, dx) → (-dy, dx)
            mem = mem.copy()
            mem[:, 0] = -mem[:, 0]
            
            tj = tj.copy()
            tj[:, 0] = -tj[:, 0]
            
            all_off = all_off.copy()
            all_off[:, 0] = -all_off[:, 0]
        
        return mem, tj, all_off
    
    def pixel_count(self):
        """Retorna cantidad total de píxeles en la célula."""
        return len(self.all_offsets())
    
    def tj_count(self):
        """Retorna cantidad de píxeles TJ."""
        return len(self.tj_offsets())
    
    def membrane_count(self):
        """Retorna cantidad de píxeles de membrana."""
        return len(self.membrane_offsets())

class Monolayer:
    """
    Monocapa de células usando un stencil compartido.
    """
    
    def __init__(self, Cellshape:str, n_cols: int, n_rows: int, resolution: int, height_width_ratio: float = 1.0, displacement: float = 0):
        """
        Args:
            n_cols: Número de columnas
            n_rows: Número de filas
            resolution: Píxeles por unidad
            height_width_ratio: Ratio alto/ancho del hexágono (1.0 = regular)
        """
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.n_cells = n_cols * n_rows
        self.Culture_indices = np.full(self.n_cells, SystemConst.ALIVE, dtype=np.int8)
        self.resolution = resolution
        self.height_width_ratio = height_width_ratio
        # UN solo stencil de referencia
        self.Cellshape = Cellshape 
        if self.Cellshape == 'hex':
            self.stencil = HexCell(resolution, self.height_width_ratio)
        elif self.Cellshape == 'rec':
            self.stencil = RecCell(resolution, self.height_width_ratio, displacement)
        elif self.Cellshape == 'tri':
            self.stencil = TriCell(resolution, self.height_width_ratio)
        
        self.centers = self.stencil.centers(n_cols, n_rows)
        self.neighbors = self.stencil.neighbors(n_cols, n_rows)

        # Estado de células (1=viva, 0=muerta)
        self.cell_states = np.ones(self.n_cells, dtype=np.int32)

    def raster_cell_type(self, map, x1, y1, side, cell_const, mem_const, tj_const):
            # === RASTERIZADO VECTORIZADO ===
            cell_indices = np.where(self.Culture_indices == cell_const)[0]
            cell_centers = self.centers[cell_indices]  # (n_alive, 2)
            
            # Coordenadas relativas al square
            cx_rel = cell_centers[:, 0] - x1  # (n_alive,)
            cy_rel = cell_centers[:, 1] - y1
            
            if self.Cellshape == 'tri':
                # Separar células por paridad de fila
                rows = cell_indices % self.n_rows
                cols = cell_indices // self.n_rows
                even_mask = ( (cols + rows) % 2 == 0)  # Pointy-up
                odd_mask = ~even_mask         # Pointy-down
                
                # Procesar pointy-up
                if np.any(even_mask):
                    mem_even, tj_even, _ = self.stencil.get_offsets(row = 0)
                    self._raster_subset(map, cx_rel[even_mask], cy_rel[even_mask], side,
                                    mem_const, tj_const,
                                    mem_even, tj_even)
                
                # Procesar pointy-down
                if np.any(odd_mask):
                    mem_odd, tj_odd, _ = self.stencil.get_offsets(row = 1)
                    self._raster_subset(map, cx_rel[odd_mask], cy_rel[odd_mask], side,
                                    mem_const, tj_const,
                                    mem_odd, tj_odd)
            else:
                mem_off, tj_off, _ = self.stencil.get_offsets( row = 0)
                # hex, rec: todos usan el mismo offset
                self._raster_subset(map, cx_rel, cy_rel, side,
                                mem_const, tj_const,
                                mem_off,
                                tj_off)
            
            return map
        
    def _raster_subset(self, map, cx_rel, cy_rel, side, mem_const, tj_const, membrane_off, tj_off):
        """Rasteriza un subconjunto de células con offsets específicos."""
        if len(cx_rel) == 0:
            return
        
        if mem_const:
            mem_px = cx_rel[:, None] + membrane_off[:, 1]
            mem_py = cy_rel[:, None] + membrane_off[:, 0]
            valid_mem = (mem_py >= 0) & (mem_py < side) & (mem_px >= 0) & (mem_px < side)
            map[mem_py[valid_mem], mem_px[valid_mem]] = mem_const
        
        # --- TJ ---
        base_tj_px = cx_rel[:, None] + tj_off[:, 1]
        base_tj_py = cy_rel[:, None] + tj_off[:, 0]
        
        if tj_const:
            tj_px = base_tj_px
            tj_py = base_tj_py
            valid_tj = (tj_py >= 0) & (tj_py < side) & (tj_px >= 0) & (tj_px < side)
            map[tj_py[valid_tj], tj_px[valid_tj]] = tj_const
        else:
            if self.Cellshape == 'rec':
                tj_px = base_tj_px.ravel()
                tj_py = base_tj_py.ravel()
            # Expandir ±1 en Y para hex y tri
            else:
                offsets = np.array([-1, 0, 1])
                tj_py = base_tj_py[:, :, None] + offsets
                tj_px = base_tj_px[:, :, None] + np.zeros(3, dtype=int)
                tj_py = tj_py.ravel()
                tj_px = tj_px.ravel()
            
            valid_tj = (tj_py >= 0) & (tj_py < side) & (tj_px >= 0) & (tj_px < side)
            
            # Solo borrar si es TJ
            is_tj = map[tj_py[valid_tj], tj_px[valid_tj]] == SystemConst.TJ
            tj_py_final = tj_py[valid_tj][is_tj]
            tj_px_final = tj_px[valid_tj][is_tj]
            
            map[tj_py_final, tj_px_final] = SystemConst.EMPTY


    def rasterize_square(self) -> dict:
        """
        Rasteriza directamente al square, sin crear mapa completo.
        
        Args:
            dead_indices: Array de índices de células muertas (opcional)
        
        Returns:
            dict con type_map, label_map, shape (ya es square)
        """

        # Bounds del square: centro primera célula a centro última
        first_center = self.centers[0]
        last_center = self.centers[-1]
        
        x1, y1 = first_center
        x2, y2 = last_center
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Hacer cuadrado (lado menor)
        rect_w = x2 - x1 + 1
        rect_h = y2 - y1 + 1
        side = min(rect_w, rect_h)
        
        # Centrar recorte
        offset_x = (rect_w - side) // 2
        offset_y = (rect_h - side) // 2
        x1 += offset_x
        y1 += offset_y
        
        # Inicializar mapas
        cult_map = np.zeros((side, side), dtype=np.int8)
        # Mapeo de estado → (mem_const, tj_const)
        type_mapping = {
            SystemConst.ALIVE:         (SystemConst.MEMBRANE,     SystemConst.TJ),
            SystemConst.FUSED:         (SystemConst.MEM_FUS,      SystemConst.TJ_FUS),
            SystemConst.INFECTED_CELL: (SystemConst.INF_MEMBRANE, SystemConst.TJ),
            SystemConst.DEAD:          (SystemConst.EMPTY,        SystemConst.EMPTY),
        }
        
        # Fixed rasterization order: ALIVE first, INFECTED second, FUSED third, DEAD last.
        # FUSED must come after ALIVE so TJ_FUS (32) overwrites TJ (8) at shared boundaries.
        # DEAD must remain last to erase the dead-cell region over whatever was painted.
        priority = [
            SystemConst.ALIVE,
            SystemConst.INFECTED_CELL,
            SystemConst.FUSED,
            SystemConst.DEAD,
        ]
        present_set = set(np.unique(self.Culture_indices).tolist())
        for cell_type in priority:
            if cell_type in present_set and cell_type in type_mapping:
                mem_const, tj_const = type_mapping[cell_type]
                cult_map = self.raster_cell_type(cult_map, x1, y1, side,
                                                 cell_type, mem_const, tj_const)

        # Pre-pass: upgrade outer TJ → TJ_FUS where adjacent to MEM_FUS.
        # Catches corner pixels the stencil rasterization does not fully cover.
        if np.any(cult_map == int(SystemConst.MEM_FUS)):
            mem_fus_pre = (cult_map == int(SystemConst.MEM_FUS))
            adj_mf_pre = np.zeros((side, side), dtype=bool)
            adj_mf_pre[1:,  :] |= mem_fus_pre[:-1, :]
            adj_mf_pre[:-1, :] |= mem_fus_pre[1:,  :]
            adj_mf_pre[:,  1:] |= mem_fus_pre[:, :-1]
            adj_mf_pre[:, :-1] |= mem_fus_pre[:, 1:]
            cult_map[(cult_map == int(SystemConst.TJ)) & adj_mf_pre] = int(SystemConst.TJ_FUS)

        # Post-rasterize pass 1: dissolve internal TJ_FUS pixels (between two fused cells → MEM_FUS)
        if np.any(cult_map == int(SystemConst.TJ_FUS)):
            fus_tj_mask  = (cult_map == int(SystemConst.TJ_FUS))
            mem_fus_mask = (cult_map == int(SystemConst.MEM_FUS))
            adj_mf = np.zeros((side, side), dtype=np.int8)
            adj_mf[1:,  :] += mem_fus_mask[:-1, :].astype(np.int8)
            adj_mf[:-1, :] += mem_fus_mask[1:,  :].astype(np.int8)
            adj_mf[:,  1:] += mem_fus_mask[:, :-1].astype(np.int8)
            adj_mf[:, :-1] += mem_fus_mask[:, 1:].astype(np.int8)
            cult_map[fus_tj_mask & (adj_mf >= 2)] = int(SystemConst.MEM_FUS)

        # Post-rasterize pass 2: upgrade TJ pixels adjacent to INF_MEMBRANE to TJ_INF_BORDER
        if np.any(cult_map == SystemConst.INF_MEMBRANE):
            inf_mask = (cult_map == int(SystemConst.INF_MEMBRANE))
            # No-wrap adjacency check (avoids false positives at map edges)
            adj_inf = np.zeros_like(inf_mask)
            adj_inf[1:,  :] |= inf_mask[:-1, :]
            adj_inf[:-1, :] |= inf_mask[1:,  :]
            adj_inf[:,  1:] |= inf_mask[:, :-1]
            adj_inf[:, :-1] |= inf_mask[:, 1:]
            cult_map[(cult_map == int(SystemConst.TJ)) & adj_inf] = int(SystemConst.TJ_INF_BORDER)

        return {
            'type_map': cult_map,
            'shape': side,
            'offset': (x1, y1),
        }
    
    def get_cell_index(self, col, row):
        return self.stencil.cell_index(col, row, self.n_rows)
    
    def compute_wound_deaths(self, radius: float, d: float, offset_x: float = 0, offset_y: float = 0):
        """Kill all cells whose physical center is within `radius` physical units of the wound center.

        radius: wound radius in physical units (same units as ac, e.g. µm).
        d: pixel size in the same units (µm/pixel). Converts radius to pixels internally.
        offset_x / offset_y shift the wound center in cell-spacing units.
        """
        radius_px = radius / d
        cx = np.mean(self.centers[:, 0]) + offset_x * self.stencil.spacing_x
        cy = np.mean(self.centers[:, 1]) + offset_y * self.stencil.spacing_y

        dist_sq = (self.centers[:, 0] - cx) ** 2 + (self.centers[:, 1] - cy) ** 2
        self.Culture_indices[dist_sq <= radius_px ** 2] = SystemConst.DEAD
        
    def compute_random_choosing(self, fraction: float, seed: int = None) -> np.ndarray:
        """Calcula índices aleatorios a elegir."""
        if seed is not None: np.random.seed(seed)
        return np.random.choice(self.n_cells, size= int(self.n_cells * fraction), replace=False)

    def compute_random_death(self, fraction: float, seed: int = None):
        """Calcula índices aleatorios a matar."""
        self.Culture_indices[self.compute_random_choosing(fraction, seed)] = SystemConst.DEAD
        return 

    def compute_random_fusion(self, fraction: float, seed: int = None):
        """Calcula índices aleatorios a fusionar."""
        self.Culture_indices[self.compute_random_choosing(fraction, seed)] = SystemConst.FUSED
        return

    def compute_random_infection(self, fraction: float, seed: int = None):
        """Marks random cells as INFECTED_CELL."""
        self.Culture_indices[self.compute_random_choosing(fraction, seed)] = SystemConst.INFECTED_CELL
        return

    def compute_colony_attachment(self, n_colonies: int, colony_radius: int, seed: int = None):
        """Colony attachment / clonogenic assay.

        Starts with all cells DEAD, seeds n_colonies random centers, then grows
        each colony outward by BFS for colony_radius hops through the neighbor
        graph.  Colonies can merge.  colony_radius=0 marks only the seed cell.
        """
        if seed is not None:
            np.random.seed(seed)
        self.Culture_indices[:] = SystemConst.DEAD
        seed_cells = np.random.choice(self.n_cells, size=min(n_colonies, self.n_cells), replace=False)
        for center in seed_cells:
            alive = {int(center)}
            frontier = {int(center)}
            for _ in range(colony_radius):
                next_frontier = set()
                for c in frontier:
                    for nb in self.neighbors[c]:
                        if nb not in alive:
                            alive.add(nb)
                            next_frontier.add(nb)
                frontier = next_frontier
                if not frontier:
                    break
            for c in alive:
                self.Culture_indices[c] = SystemConst.ALIVE
    
    def get_cell_pixels_abs(self, raster, cell_index):
        cx, cy = self.centers[cell_index]
        ox, oy = raster['offset']
        base = np.array([cy - oy, cx - ox])

        mem = self.stencil.membrane_offsets() + base
        tj  = self.stencil.tj_offsets() + base

        return mem, tj

    def cell_stats(self):
        """Estadísticas del cultivo."""
        return {
            'total_cells': self.n_cells,
            'alive': np.sum(self.cell_states == 1),
            'dead': np.sum(self.cell_states == 0),
            'pixels_per_cell': self.stencil.pixel_count(),
            'tj_per_cell': self.stencil.tj_count(),
            'membrane_per_cell': self.stencil.membrane_count()
        }
    
    def infect_cell(self, raster, cell_index):
        type_map = raster['type_map']
        mem, tj = self.get_cell_pixels_abs(raster, cell_index)

        # Bounds
        h, w = type_map.shape
        valid_mem = (mem[:,0]>=0)&(mem[:,0]<h)&(mem[:,1]>=0)&(mem[:,1]<w)
        valid_tj  = (tj[:,0]>=0)&(tj[:,0]<h)&(tj[:,1]>=0)&(tj[:,1]<w)

        # TJ -> MEMBRANE
        mask_tj = valid_tj & (type_map[tj[:,0], tj[:,1]] == SystemConst.TJ)
        type_map[tj[mask_tj,0], tj[mask_tj,1]] = SystemConst.FUSED

        # MEMBRANE -> INFECTED
        mask_mem = valid_mem & (type_map[mem[:,0], mem[:,1]] == SystemConst.MEMBRANE)
        type_map[mem[mask_mem,0], mem[mask_mem,1]] = SystemConst.FUSED
    
    def fuse_cells_auto(
        self,
        raster: dict,
        seed_cells: np.ndarray,
        propagate: bool = True
    ):
        """
        Fusiones múltiples automáticas con infección.
        """
        fused = set()
        queue = list(seed_cells)

        while queue:
            c = queue.pop(0)
            if c in fused:
                continue
            fused.add(c)
            # Infectar célula actual
            self.infect_cell(raster, c)

            if propagate:
                for nb in self.neighbors[c]:
                    if nb not in fused:
                        queue.append(nb)

        return np.array(sorted(fused), dtype=np.int32)

def construct_culture(
    Cellshape: str,
    n_cols: int,
    n_rows: int,
    n: int,
    simulation_type: list = ["confluent"],
    percentage: float = 0.0,
    wound_rad: float = 0.0,
    seed: int = 0,
    height_width_ratio: float = 1.0,
    displacement: float = 0,
    n_colonies: int = 1,
    colony_radius: int = 3,
    ac: float = 0.0,
) -> Tuple[np.ndarray, dict]:
    """
    Versión optimizada: calcula muertes primero, rasteriza una vez.
    """
    monolayer = Monolayer(Cellshape, n_cols, n_rows, n, height_width_ratio, displacement)

    # Pixel size: derived from cell area and stencil pixel count (same units as ac)
    cell_pixel_count = monolayer.stencil.membrane_count() + monolayer.stencil.tj_count() / 2
    d = np.sqrt(ac / cell_pixel_count) if ac > 0 else 1.0

    # 1. CALCULAR MUERTES (solo índices, O(1) en memoria de mapa)
    if isinstance(simulation_type, str):
        simulation_type = [simulation_type]
    # Aplicar cada tipo de simulación
    for sim in simulation_type:
        if sim == "wound" and wound_rad >= 0:
            monolayer.compute_wound_deaths(radius=wound_rad, d=d)
        
        elif sim == "random" and percentage > 0:
            monolayer.compute_random_death(percentage, seed=seed)
        
        elif sim == "fused" and percentage > 0:
            monolayer.compute_random_fusion(percentage, seed=seed)

        elif (sim == "infected" or sim == "inf") and percentage > 0:
            monolayer.compute_random_infection(percentage, seed=seed)

        elif sim == "colony":
            monolayer.compute_colony_attachment(n_colonies, colony_radius, seed=seed)


    # 2. RASTERIZAR UNA VEZ (directo a square, con muertes aplicadas)
    square = monolayer.rasterize_square()
    
    side = square['shape']
    type_map = square['type_map']
    
    stats = {
        'dim': side **2,
        'side': side,
        'membrane_per_cell': monolayer.stencil.membrane_count(),
        'tj_per_cell': monolayer.stencil.tj_count(),
        'pixels_per_cell': monolayer.stencil.pixel_count(),
        'total_membrane': np.sum(type_map == SystemConst.MEMBRANE),
        'total_tj': np.sum(type_map == SystemConst.TJ),
        'total_fused': np.sum(type_map == SystemConst.MEM_FUS),
        'total_infected': np.sum(type_map == SystemConst.INF_MEMBRANE),
        'total_tj_inf_border': np.sum(type_map == SystemConst.TJ_INF_BORDER),
        #'alive': np.sum(monolayer.Culture_indices == SystemConst.ALIVE),
        #'dead': np.sum(monolayer.Culture_indices == SystemConst.EMPTY),
        #'fused': np.sum(monolayer.Culture_indices == SystemConst.FUSED)
    }
    
    return type_map.ravel(order='F'), stats


def compute_rb_rescale_map(
    culture_vector: np.ndarray,
    side: int,
    ref_tj_count: int,
    normalization: str = 'perimeter',
) -> np.ndarray:
    """
    Compute per-pixel Rb scaling factors for fused islands.

    After rasterize_square() + Pass 1 (internal TJ dissolved), every remaining
    TJ_FUS pixel is on the outer boundary of a fused island.  This function
    labels connected MEM_FUS regions, measures each island's outer TJ_FUS pixel
    count (pixel perimeter), and returns a float array with:

        scale_factor = pixel_perimeter / ref_tj_count   (normalization='perimeter')
        scale_factor = sqrt(N_pixels_island)            (normalization='surface')

    Non-TJ_FUS pixels are 0.  In SolveMatrix, TJ_FUS pixel i gets admittance
    c_rb / scale_factor[i] instead of the uniform c_rb.

    Args:
        culture_vector: flattened Culture_vector (Fortran order), shape (side*side,)
        side:           pixel side length of the square domain
        ref_tj_count:   TJ pixel count of a single isolated cell (cell.tj_count())
        normalization:  'perimeter' (default) or 'surface'

    Returns:
        contour_mask: float array shape (side*side,), Fortran order
    """
    type_map = culture_vector.reshape(side, side, order='F')
    mem_fus_mask = (type_map == int(SystemConst.MEM_FUS))
    contour_mask = np.zeros(side * side, dtype=float)

    if not np.any(mem_fus_mask) or ref_tj_count == 0:
        return contour_mask

    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    labeled, n_islands = _scipy_label(mem_fus_mask, structure=kernel)

    tj_fus_mask = (type_map == int(SystemConst.TJ_FUS))

    for i in range(1, n_islands + 1):
        island_mask = (labeled == i)

        # Outer TJ_FUS pixels: TJ_FUS pixels adjacent to this island
        adj = np.zeros_like(island_mask)
        adj[1:,  :] |= island_mask[:-1, :]
        adj[:-1, :] |= island_mask[1:,  :]
        adj[:,  1:] |= island_mask[:, :-1]
        adj[:, :-1] |= island_mask[:, 1:]
        outer_tj = tj_fus_mask & adj

        pixel_perimeter = int(np.sum(outer_tj))
        if pixel_perimeter == 0:
            continue

        if normalization == 'surface':
            scale = float(np.sqrt(np.sum(island_mask)))
        else:  # perimeter
            scale = pixel_perimeter / ref_tj_count

        contour_mask[outer_tj.ravel(order='F')] = scale

    return contour_mask