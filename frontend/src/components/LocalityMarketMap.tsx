import { useMemo } from 'react';
import { CircleMarker, MapContainer, Popup, TileLayer, Tooltip } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import type { LocalityBenchmark } from '../types';

/**
 * Pune Locality Market Map — real geographic visualization.
 *
 * Data rule: every price shown comes from the `benchmarks` prop (served
 * backend benchmark calibration). No benchmark values are hardcoded here.
 *
 * Coordinates: REAL geographic (WGS84) locality centers for the 12
 * supported Pune localities, used as visualization metadata only.
 * Markers represent locality centers, not individual properties, and
 * must never be presented as property-level locations (see the
 * on-screen disclaimer).
 */
const LOCALITY_COORDS: Record<string, [number, number]> = {
  Hinjawadi: [18.5917, 73.738],
  Wakad: [18.5992, 73.7767],
  Punawale: [18.6238, 73.7279],
  Ravet: [18.644, 73.7515],
  Balewadi: [18.5649, 73.7732],
  Baner: [18.5595, 73.786],
  Aundh: [18.5589, 73.8048],
  Kothrud: [18.5074, 73.8077],
  'Viman Nagar': [18.5679, 73.9143],
  Kharadi: [18.5317, 73.9419],
  Wagholi: [18.5787, 73.9837],
  Hadapsar: [18.508, 73.926],
};

const PUNE_CENTER: [number, number] = [18.575, 73.845];
const PUNE_ZOOM = 11;

export default function LocalityMarketMap({
  benchmarks,
  selected,
  onSelect,
  formatRate,
}: {
  benchmarks: LocalityBenchmark[];
  selected: string | null;
  onSelect?: (location: string) => void;
  formatRate: (v: number | null | undefined) => string;
}) {
  const points = useMemo(() => {
    const rates = benchmarks.map((b) => b.benchmark_price_per_sqft);
    const min = Math.min(...rates);
    const max = Math.max(...rates);
    const span = max - min || 1;
    return benchmarks
      .filter((b) => LOCALITY_COORDS[b.location])
      .map((b) => ({
        ...b,
        pos: LOCALITY_COORDS[b.location],
        t: (b.benchmark_price_per_sqft - min) / span,
      }))
      .sort((a, b) => a.location.localeCompare(b.location));
  }, [benchmarks]);

  const active = useMemo(
    () => points.find((p) => p.location === selected) ?? null,
    [points, selected],
  );

  const rates = points.map((p) => p.benchmark_price_per_sqft);
  const minRate = rates.length ? Math.min(...rates) : null;
  const maxRate = rates.length ? Math.max(...rates) : null;

  if (points.length === 0) {
    return <div className="text-[12px] text-aa-muted">Locality benchmarks unavailable.</div>;
  }

  return (
    <div>
      <div className="text-[10px] tracking-[0.08em] font-semibold text-aa-faint">
        PUNE LOCALITY MARKET CONTEXT
      </div>
      <div className="mt-2 rounded-lg border border-aa-border overflow-hidden h-[340px] lg:h-[440px]">
        <MapContainer
          center={PUNE_CENTER}
          zoom={PUNE_ZOOM}
          scrollWheelZoom={false}
          className="aa-leaflet"
          style={{ height: '100%', width: '100%' }}
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          />
          {points.map((p) => {
            const isSelected = p.location === selected;
            return (
              <CircleMarker
                key={p.location}
                center={p.pos}
                radius={9 + p.t * 9}
                pathOptions={{
                  color: isSelected ? '#dbeafe' : '#93c5fd',
                  weight: isSelected ? 3 : 1.5,
                  fillColor: '#3b82f6',
                  fillOpacity: 0.3 + 0.5 * p.t,
                }}
                eventHandlers={{ click: () => onSelect?.(p.location) }}
              >
                {isSelected && (
                  <Tooltip permanent direction="top" offset={[0, -14]} className="aa-map-tip">
                    {p.location}
                  </Tooltip>
                )}
                {!isSelected && <Tooltip direction="top">{p.location}</Tooltip>}
                <Popup>
                  <div className="aa-map-popup">
                    <div className="aa-map-popup-title">{p.location}</div>
                    <div className="aa-map-popup-rate">
                      {formatRate(p.benchmark_price_per_sqft)}
                      <span className="aa-map-popup-unit"> / sq ft</span>
                    </div>
                    <div className="aa-map-popup-note">
                      {p.calibration_note || p.benchmark_source}
                      {isSelected ? ' · Selected' : ''}
                    </div>
                  </div>
                </Popup>
              </CircleMarker>
            );
          })}
          {active && (
            <CircleMarker
              center={active.pos}
              radius={9 + active.t * 9 + 6}
              pathOptions={{
                color: '#60a5fa',
                weight: 1.5,
                dashArray: '4 3',
                fill: false,
              }}
              interactive={false}
            />
          )}
        </MapContainer>
      </div>
      <div className="flex items-center gap-2 mt-2">
        <span className="text-[10px] text-aa-faint">Lower benchmark</span>
        <div
          className="h-1.5 flex-1 rounded-full"
          style={{ background: 'linear-gradient(90deg, rgba(59,130,246,0.10), rgba(59,130,246,0.65))' }}
        />
        <span className="text-[10px] text-aa-faint">Higher benchmark</span>
        <span className="aa-mono text-[10px] text-aa-muted ml-1">
          {formatRate(minRate)} – {formatRate(maxRate)}/sq ft
        </span>
      </div>
      <div className="mt-2 rounded-lg border border-aa-border bg-[#0a0e1a] p-3">
        {active ? (
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
            <div className="text-[14px] font-semibold text-white">{active.location}</div>
            <div className="aa-mono text-[15px] font-bold text-white">
              {formatRate(active.benchmark_price_per_sqft)}
              <span className="text-[11px] font-normal text-aa-muted"> / sq ft</span>
            </div>
            <span className="aa-mono text-[10px] px-1.5 py-[2px] rounded bg-aa-blue/15 border border-aa-blue/30 text-aa-blueText">
              ● Selected
            </span>
            <div className="w-full text-[10.5px] text-aa-faint">
              {active.calibration_note || active.benchmark_source}
            </div>
          </div>
        ) : (
          <div className="text-[12px] text-aa-muted">
            Select a locality in the prediction form, or click a marker, to highlight it here.
          </div>
        )}
      </div>
      <div className="pt-2 text-[10.5px] text-aa-faint">
        Locality-level market visualization; markers represent locality centers, not individual properties.
      </div>
    </div>
  );
}
