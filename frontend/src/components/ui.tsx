import type { ReactNode } from 'react';

export function StatusBadge({ tone, children }: { tone: 'green' | 'red' | 'amber' | 'blue' | 'gray'; children: ReactNode }) {
  const map: Record<string, string> = {
    green: 'bg-aa-greenSoft text-green-400 border-green-500/30',
    red: 'bg-red-500/10 text-red-400 border-red-500/30',
    amber: 'bg-aa-amberSoft text-amber-400 border-amber-500/30',
    blue: 'bg-blue-500/10 text-aa-blueText border-blue-500/30',
    gray: 'bg-white/5 text-aa-muted border-aa-border',
  };
  return (
    <span className={`aa-badge border ${map[tone]}`}>
      <span className="inline-block h-1.5 w-1.5 rounded-full bg-current" />
      {children}
    </span>
  );
}

export function MetricCard({
  label,
  topRight,
  children,
}: {
  label: string;
  topRight?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="aa-card p-3.5 min-h-[118px] flex flex-col">
      <div className="flex items-center justify-between">
        <div className="text-[10px] tracking-[0.08em] font-semibold text-aa-faint uppercase">{label}</div>
        {topRight && <div className="aa-mono text-[10.5px] text-aa-faint">{topRight}</div>}
      </div>
      <div className="mt-2 flex-1">{children}</div>
    </div>
  );
}

export function SectionCard({
  title,
  subtitle,
  right,
  children,
  className = '',
}: {
  title: string;
  subtitle?: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`aa-card ${className}`}>
      <div className="flex items-start justify-between px-4 pt-3.5 pb-3 border-b border-aa-borderSoft">
        <div>
          <div className="text-[13px] font-semibold text-aa-text flex items-center gap-2">
            <span className="text-aa-blueText">◈</span> {title}
          </div>
          {subtitle && <div className="aa-mono text-[11px] text-aa-faint mt-0.5">{subtitle}</div>}
        </div>
        {right && <div className="flex items-center gap-2">{right}</div>}
      </div>
      <div className="p-4">{children}</div>
    </div>
  );
}

export function PageHeader({
  title,
  desc,
  right,
}: {
  title: string;
  desc: string;
  right?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-3 mb-4">
      <div>
        <h1 className="text-[20px] font-bold text-white leading-tight">{title}</h1>
        <p className="text-[12.5px] text-aa-muted mt-1">{desc}</p>
      </div>
      {right && <div className="flex items-center gap-2">{right}</div>}
    </div>
  );
}

export function EmptyState({ title, desc, action }: { title: string; desc?: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center py-10 text-center">
      <div className="h-10 w-10 rounded-lg bg-white/5 border border-aa-border flex items-center justify-center text-aa-faint text-lg">
        ◈
      </div>
      <div className="mt-3 text-[12.5px] text-aa-muted">{title}</div>
      {desc && <div className="mt-1 text-[11.5px] text-aa-faint">{desc}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function KeyVal({ k, v, mono = false }: { k: string; v: ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between py-[7px] border-b border-white/[0.04] last:border-0">
      <span className="text-[12px] text-aa-muted">{k}</span>
      <span className={`text-[12px] text-aa-text ${mono ? 'aa-mono' : ''}`}>{v}</span>
    </div>
  );
}

export function InfoPanel({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-lg border border-aa-border bg-[#0b1322] px-3.5 py-3 text-[11.5px] leading-relaxed text-aa-muted flex gap-2.5">
      <span className="text-aa-blueText mt-[1px]">ⓘ</span>
      <div>{children}</div>
    </div>
  );
}
