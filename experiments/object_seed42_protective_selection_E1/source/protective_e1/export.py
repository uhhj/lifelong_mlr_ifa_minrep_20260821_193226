"""Run the unchanged E0 feature exporter with package binding only."""
from .bindings import bind
bind()
from libero.lifelong.save_feature import main
if __name__ == '__main__':
    main()
